"""Generate text from a trained WangGPT model.

Usage:
    uv run chat.py --model models/with_pe_best.pt --prompt "luke skywalker"
    uv run chat.py --model models/with_pe_best.pt --prompts-file prompts.txt
    uv run chat.py --model models/with_pe_best.pt --prompt "the force" --max-tokens 50
"""

import argparse
import os
import re
from typing import Any, Optional

import torch

from utils import Tokenizer, translate_state_dict
from wang_gpt import Config, WangGPT


def load_checkpoint(model_path: str) -> tuple[WangGPT, Tokenizer, str]:
    """Loads a WangGPT model and tokenizer from a checkpoint.

    Handles both the current dict format (with config and state_dict)
    and the legacy format (raw state_dict with hyperparams in the filename).

    Args:
    - model_path: Path to the .pt checkpoint file.

    Returns:
    - (model, tokenizer, device)
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.load("models/unified_vocab.pt")
    checkpoint: Any = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config_dict: dict[str, Any] = checkpoint["config"]
        state_dict: dict[str, torch.Tensor] = checkpoint["model_state_dict"]
        config = Config(
            d_vocab=tokenizer.vocab_size,
            d_model=config_dict["d_m"],
            n_layers=config_dict.get("layers", 7),
            n_heads=4,
            window_size=config_dict["window_size"],
            use_pos_emb=config_dict.get("use_pe", True),
        )
    else:
        # Legacy format: reconstruct config from filename
        print(f"Legacy checkpoint detected, reconstructing config...")
        filename = os.path.basename(model_path)
        dm_match: Optional[re.Match] = re.search(r"dm(\d+)", filename)
        d_m = int(dm_match.group(1)) if dm_match else 120
        config = Config(
            d_vocab=tokenizer.vocab_size,
            d_model=d_m,
            n_layers=7,
            n_heads=4,
            window_size=128,
        )
        state_dict = translate_state_dict(checkpoint, d_m)

    model = WangGPT(config)

    # Strip DataParallel "module." prefix if present
    cleaned_sd = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned_sd, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained WangGPT model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--prompt", help="Text prompt for generation")
    parser.add_argument("--prompts-file", help="File with one prompt per line")
    parser.add_argument("--max-tokens", type=int, default=30, help="Tokens to generate (default: 30)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold (default: 0.9)")
    args = parser.parse_args()

    if not args.prompt and not args.prompts_file:
        parser.error("Provide either --prompt or --prompts-file")

    model, tokenizer, device = load_checkpoint(args.model)
    print(f"Loaded {os.path.basename(args.model)} on {device}")

    # Collect prompts
    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts.extend(line.strip() for line in f if line.strip())

    for prompt in prompts:
        output = model.generate_text(
            prompt,
            tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")


if __name__ == "__main__":
    main()
