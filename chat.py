import glob
import os
import re
from typing import Any

import torch
import torch.nn.functional as F
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

from utils import clean_tokenization,\
    TOK_TO_IDX, IDX_TO_TOK, VOCAB_SIZE
from wang_gpt import WangGPT

CONSOLE = Console()


def load_checkpoint(model_path: str):
    """Loads a Star Wars DIY-GPT model from a saved checkpoint.

    Handles both new dictionary checkpoints (containing config and state_dict)
    and legacy state_dict files by attempting to reconstruct hyperparameters.

    Args:
    - model_path: Path to the .pt checkpoint file.

    Returns:
    - A tuple (model, config, device).

    Raises:
    - RuntimeError: If the model architecture doesn't match the checkpoint.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)

    # Check if it's the new dictionary format or just the state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
    else:
        # Try to reconstruct from filename and old format data
        rprint("[yellow]⚠️  Ancient Holocron detected (Legacy format). Deciphering...[/yellow]")
        state_dict = checkpoint

        # Parse filename: eps1_3-lr0.0003-bs32-dm236.pt
        filename = os.path.basename(model_path)
        dm_match = re.search(r"dm(\d+)", filename)
        bs_match = re.search(r"bs(\d+)", filename)

        d_m = int(dm_match.group(1)) if dm_match else 252
        batch_size = int(bs_match.group(1)) if bs_match else 32

        # Detect if this was before or after the d_ff fix
        # Check shape of TBs.0.Wff_in which is (d_m, d_ff)
        first_layer_key = "TBs.0.Wff_in"
        if "module.TBs.0.Wff_in" in state_dict:
            first_layer_key = "module.TBs.0.Wff_in"

        d_ff = state_dict[first_layer_key].shape[1]

        config = {
            "d_m": d_m,
            "layers": 7,
            "window_size": 128,
            "batch_size": batch_size,
            "d_ff_override": d_ff # We'll need to pass this to WangGPT
        }
        rprint(f"[green]Reconstructed: d_m={d_m}, d_ff={d_ff}, vocab={VOCAB_SIZE}[/green]")

    # Initialize model
    model = WangGPT(
        d_v=VOCAB_SIZE,
        d_m=config["d_m"],
        num_tbs=config["layers"],
        w=config["window_size"],
        b=config.get("batch_size", 32),
        use_pe=config.get("use_pe", True) # Default to True for old models
    )

    # If the d_ff in the model doesn't match our guess, adjust it
    if "d_ff_override" in config:
        for tb in model.TBs:
            import torch.nn as nn
            tb.d_ff = config["d_ff_override"]
            tb.Wff_in = nn.Parameter(torch.randn(config["d_m"], tb.d_ff) * 0.02)
            tb.Wff_out = nn.Parameter(torch.randn(tb.d_ff, config["d_m"]) * 0.02)

    # Handle DataParallel prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        rprint("[bold red]Critical mismatch in neural pathways![/bold red]")
        rprint(str(e))
        raise e

    model.to(device)
    model.eval()

    return model, config, device


def top_k_sampling(logits: torch.tensor, k: int = 10, temperature: float = 1.0):
    """Filters logits to keep the top-k values and samples from the softmax.

    Args:
    - logits: 1D tensor of predicted log probabilities.
    - k: The number of highest probability tokens to consider.
    - temperature: Scaling factor (higher means more random).

    Returns:
    - The selected token index (int).
    """

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    v, i = torch.topk(logits, k)
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(0, i, v)

    # Softmax to get probabilities
    probs = F.softmax(out, dim=-1)

    # Sample
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate_response(model: torch.nn.Module, config: dict[str, Any],
                      device: str, prompt_text: str, max_new_tokens: int = 30,
                      temperature: float = 0.7, top_k: int = 10):
    """Generates a next-token prediction sequence based on a user prompt.

    Args:
    - model: The trained WangGPT model instance.
    - config: The model configuration dictionary.
    - device: The torch device (cuda/cpu).
    - prompt_text: The user-supplied input string.
    - max_new_tokens: The number of tokens to generate.
    - temperature: Sampling temperature.
    - top_k: Top-k sampling limit.

    Returns:
    - The generated response string.
    """

    tokens = clean_tokenization(prompt_text)
    if not tokens:
        return "The Force is silent. Please try another prompt."

    # Filter tokens that are in the vocab
    indices = [TOK_TO_IDX[t] for t in tokens if t in TOK_TO_IDX]
    if not indices:
        return "I do not recognize those terms in my data banks."

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr = indices[-config["window_size"]:]
            curr_tensor = torch.tensor([curr], dtype=torch.long).to(device)

            logits = model(curr_tensor)
            last_logits = logits[0, -1, :]

            next_idx = top_k_sampling(last_logits, k=top_k, temperature=temperature)
            indices.append(next_idx)

    response_words = [IDX_TO_TOK[i] for i in indices[len(tokens):]]
    return " ".join(response_words)


def main():
    """Launches the Star Wars DIY-GPT Chat CLI.

    Displays a selection of available models, loads the chosen holocron,
    and enters a conversation loop with the user.
    """

    CONSOLE.clear()
    welcome_text = """[bold blue]🌌 Star Wars DIY-GPT Chat CLI 🌌[/bold blue]
[italic]Your own personal droid is ready to converse.[/italic]"""
    CONSOLE.print(Panel.fit(welcome_text, border_style="cyan"))

    models = sorted(glob.glob("models/*.pt"))
    if not models:
        rprint("[bold red]Error: No models found. Run training first![/bold red]")
        return

    table = Table(title="Available Holocrons")
    table.add_column("ID", justify="center", style="cyan")
    table.add_column("Model Name", style="magenta")
    for i, m in enumerate(models):
        table.add_row(str(i), os.path.basename(m))

    CONSOLE.print(table)
    m_choice = IntPrompt.ask("Choose a Holocron", choices=[str(i) for i in range(len(models))], default=0)

    with CONSOLE.status(f"[bold green]Syncing with {os.path.basename(models[m_choice])}...[/bold green]"):
        try:
            model, config, device = load_checkpoint(models[m_choice])
        except Exception as e:
            rprint(f"[bold red]Link failure: {e}[/bold red]")
            return

    rprint(f"\n[bold green]✅ Connection Established.[/bold green] Mode: [bold cyan]{device}[/bold cyan]")
    rprint("[italic]Type 'exit' to terminate.[/italic]\n")

    while True:
        u_input = Prompt.ask("[bold yellow]User[/bold yellow]")
        if u_input.lower() in ["exit", "quit"]:
            rprint("\n[bold cyan]May the Force be with you.[/bold cyan]")
            break

        with CONSOLE.status("[italic]Analyzing...[/italic]"):
            response = generate_response(model, config, device, u_input)

        CONSOLE.print(Panel(response, title="[bold blue]DIY-GPT[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    main()
