import glob
import os
import re
from typing import Any, Optional

import torch
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

from utils import clean_tokenization, translate_state_dict,\
    TOK_TO_IDX, IDX_TO_TOK, VOCAB_SIZE
from wang_gpt import WangGPT, Config

CONSOLE = Console()


def load_checkpoint(model_path: str) -> tuple[WangGPT, Config, str]:
    """Loads a Star Wars DIY-GPT model from a saved checkpoint.

    Handles both new dictionary checkpoints (containing config and state_dict)
    and legacy state_dict files by attempting to reconstruct hyperparameters
    and mapping weight keys to the new architecture.

    Args:
    - model_path: Path to the .pt checkpoint file.

    Returns:
    - A tuple (model, config, device).

    Raises:
    - RuntimeError: If the model architecture doesn't match the checkpoint.
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint: Any = torch.load(model_path, map_location=device)

    # Check if it's the new dictionary format or just the state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config_dict: dict[str, Any] = checkpoint["config"]
        state_dict: dict[str, torch.Tensor] = checkpoint["model_state_dict"]
        # Convert dict to Config object if it's not already
        config: Config = Config(
            d_vocab=VOCAB_SIZE,
            d_model=config_dict["d_m"],
            n_layers=config_dict["layers"],
            n_heads=4,
            window_size=config_dict["window_size"]
        )
    else:
        # Try to reconstruct from filename and old format data
        rprint("[yellow]⚠️  Ancient Holocron detected (Legacy format). "
               "Deciphering...[/yellow]")
        state_dict_raw: dict[str, torch.Tensor] = checkpoint

        # Parse filename for dm and batch_size
        filename: str = os.path.basename(model_path)
        dm_match: Optional[re.Match] = re.search(r"dm(\d+)", filename)
        d_m: int = int(dm_match.group(1)) if dm_match else 120

        config = Config(
            d_vocab=VOCAB_SIZE,
            d_model=d_m,
            n_layers=7,
            n_heads=4,
            window_size=128
        )

        # Translate the legacy state dict
        state_dict = translate_state_dict(state_dict_raw, d_m)
        rprint(f"[green]Reconstructed: d_m={d_m}, layers=7[/green]")

    # Initialize model
    model: WangGPT = WangGPT(config)

    # Handle DataParallel prefix if present
    new_state_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        rprint("[bold red]Critical mismatch in neural pathways![/bold red]")
        rprint(str(e))
        raise e

    model.to(device)
    model.eval()

    return model, config, device


def generate_response(model: WangGPT, device: str, prompt_text: str,
                      max_new_tokens: int = 30, temperature: float = 0.7,
                      top_k: int = 50, top_p: float = 0.9) -> str:
    """Generates a next-token prediction sequence based on a user prompt.

    Uses the model's built-in generate method with advanced sampling.

    Args:
    - model: The trained WangGPT model instance.
    - device: The torch device (cuda/cpu).
    - prompt_text: The user-supplied input string.
    - max_new_tokens: The number of tokens to generate.
    - temperature: Sampling temperature.
    - top_k: Top-k sampling limit.
    - top_p: Top-p (nucleus) sampling threshold.

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

    idx = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        generated_idx = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    response_words = [IDX_TO_TOK[i.item()] for i in generated_idx[0][len(indices):]]
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

    # Filter out unified_vocab.pt and other non-model files
    models = sorted([m for m in glob.glob("models/*.pt") if "unified_vocab"
                     not in m])

    if not models:
        rprint("[bold red]Error: No models found. Run training first!"
               "[/bold red]")
        return

    def select_and_load_model():
        """Prompts the user to select a model and loads it."""
        table = Table(title="Available Holocrons")
        table.add_column("ID", justify="center", style="cyan")
        table.add_column("Model Name", style="magenta")
        for i, m in enumerate(models):
            table.add_row(str(i), os.path.basename(m))

        CONSOLE.print(table)
        m_choice = IntPrompt.ask("Choose a Holocron",
                                 choices=[str(i) for i in range(len(models))],
                                 default=0)

        with CONSOLE.status(f"[bold green]Syncing with {os.path.basename(
            models[m_choice])}...[/bold green]"):
            try:
                model, _, device = load_checkpoint(models[m_choice])
                return model, device, os.path.basename(models[m_choice])
            except Exception as e:
                rprint(f"[bold red]Link failure: {e}[/bold red]")
                return None, None, None

    model, device, model_name = select_and_load_model()
    if model is None:
        return

    rprint(f"\n[bold green]✅ Connection Established with {model_name}."
           "[/bold green] Mode: [bold cyan]{device}[/bold cyan]")
    rprint("[italic]Type '/exit' to terminate or '/switch' to change "
           "models.[/italic]\n")

    while True:
        u_input = Prompt.ask("[bold yellow]User[/bold yellow]")

        if u_input.lower() == "/exit":
            rprint("\n[bold cyan]May the Force be with you.[/bold cyan]")
            break

        if u_input.lower() == "/switch":
            new_data = select_and_load_model()
            if new_data[0]:
                model, device, model_name = new_data
                rprint(f"\n[bold green]✅ Switched to {model_name}."
                       "[/bold green]")
            continue

        if u_input.lower() in ["exit", "quit"]:
            rprint("[yellow]Hint: Use '/exit' to terminate the "
                   "session.[/yellow]")
            continue

        with CONSOLE.status("[italic]Analyzing...[/italic]"):
            response = generate_response(model, device, u_input)

        CONSOLE.print(Panel(response, title=f"[bold blue]DIY-GPT ({model_name})"
                            "[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    main()
