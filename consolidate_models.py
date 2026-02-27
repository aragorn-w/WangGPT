import glob
import os
import re

import torch
import torch.nn as nn

from utils import clean_tokenization,\
    UNIFIED_VOCAB, TOK_TO_IDX, VOCAB_SIZE
from wang_gpt import WangGPT


def evaluate_model(model: torch.nn.Module, data: torch.tensor,
                   win_size: int, batch_size: int, device: str):
    """Evaluates a model's performance on a validation dataset.

    Calculates the average cross-entropy loss over a subset of the
    validation data to determine model quality.

    Args:
    - model: The WangGPT model to evaluate.
    - data: Validation dataset tensor of token indices.
    - win_size: Context window size.
    - batch_size: Evaluation batch size.
    - device: Torch device (cuda/cpu).

    Returns:
    - Float representing the average loss.
    """

    model.eval()
    loss_func = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        max_idx = len(data) - win_size - 1
        if max_idx <= 0: return float("inf")

        steps = min(50, max_idx // batch_size + 1) if max_idx > batch_size else 1
        for i in range(steps):
            start_pos = (i * batch_size) % max_idx
            indices = [(start_pos + j) % max_idx for j in range(batch_size)]
            x = torch.stack([data[idx: idx + win_size] for idx in indices]).to(device)
            y = torch.stack([data[idx + 1: idx + 1 + win_size] for idx in indices]).to(device)

            logits = model(x)
            loss = loss_func(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")


def parse_filename(filename: str):
    """Extracts model hyperparameters from a standard result filename.

    Args:
    - filename: The .pt filename (e.g., 'eps1_3-lr0.0003-dm252.pt').

    Returns:
    - Dictionary containing d_m, lr, batch_size, layers, and window_size.
    """

    # Example: eps1_3-lr0.0003-bs32-dm236.pt
    dm_match = re.search(r"-dm(\d+)", filename)
    dm = int(dm_match.group(1)) if dm_match else 252

    lr_match = re.search(r"-lr([\d\.]+)", filename)
    lr = float(lr_match.group(1)) if lr_match else 0.0003

    bs_match = re.search(r"-bs(\d+)", filename)
    bs = int(bs_match.group(1)) if bs_match else 32

    return {"d_m": dm, "lr": lr, "batch_size": bs, "layers": 7, "window_size": 128}


def consolidate():
    """Finds the best-performing model for each episode and saves it as a final holocron.

    Iterates through all models in the models/ directory, evaluates them on
    validation data, saves the best one with its metadata, and removes the rest.
    """

    model_files = glob.glob("models/*.pt")
    # Exclude already consolidated best models and the unified vocab
    model_files = [f for f in model_files if not f.endswith("_best.pt") and f != "models/unified_vocab.pt"]

    if not model_files:
        print("No new models found to consolidate.")
        return

    groups = {}
    for f in model_files:
        base = os.path.basename(f).split("-")[0]
        if base not in groups:
            groups[base] = []
        groups[base].append(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for episode, files in groups.items():
        print(f"\n--- Consolidating {episode} ({len(files)} models) ---")

        # Try to find the data file. Fall back to combined_star_wars.txt
        data_path = f"data/{episode}.txt"
        if not os.path.exists(data_path):
            data_path = "data/combined_star_wars.txt"
            print(f"Data file {episode}.txt not found. Falling back to {data_path}")

        if not os.path.exists(data_path):
            print(f"No data source found for {episode}. Skipping.")
            continue

        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        all_tokens = clean_tokenization(text)

        split_idx = int(len(all_tokens) * 0.8)
        val_tokens = all_tokens[split_idx:]
        val_data = torch.tensor([TOK_TO_IDX[w] for w in val_tokens
                                 if w in TOK_TO_IDX], dtype=torch.long)

        best_loss = float("inf")
        best_model_path = None

        for model_path in files:
            print(f"Evaluating {model_path}...", end=" ", flush=True)
            try:
                checkpoint = torch.load(model_path, map_location=device)

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    config = checkpoint["config"]
                    state_dict = checkpoint["model_state_dict"]
                else:
                    config = parse_filename(os.path.basename(model_path))
                    state_dict = checkpoint

                model = WangGPT(
                    d_v=VOCAB_SIZE,
                    d_m=config["d_m"],
                    num_tbs=config["layers"],
                    w=config["window_size"],
                    b=config["batch_size"],
                    use_pe=config.get("use_pe", True)
                )

                # Check if vocab size matches
                if state_dict["U"].shape[1] != VOCAB_SIZE:
                    print(f"Vocab mismatch! Expected {state_dict['U'].shape[1]}"
                          f", got {VOCAB_SIZE}. Skipping.")
                    continue

                # Handle DataParallel prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                model.load_state_dict(new_state_dict)
                model.to(device)

                loss = evaluate_model(model, val_data, config["window_size"], config["batch_size"], device)
                print(f"Loss: {loss:.4f}")

                if loss < best_loss:
                    best_loss = loss
                    best_model_path = model_path
            except Exception as e:
                print(f"Error: {e}")

        if best_model_path:
            final_name = f"models/{episode}_best.pt"
            print(f"Best model for {episode} is {best_model_path} with loss {best_loss:.4f}")
            # Keep the best one, delete others
            best_ckpt = torch.load(best_model_path, map_location="cpu")
            if not (isinstance(best_ckpt, dict) and "model_state_dict" in best_ckpt):
                # Upgrade it to a full checkpoint
                best_ckpt = {
                    "model_state_dict": best_ckpt,
                    "config": parse_filename(os.path.basename(best_model_path)),
                    "vocab": UNIFIED_VOCAB,
                    "run_name": os.path.basename(best_model_path).replace(".pt", "")
                }
            torch.save(best_ckpt, final_name)
            print(f"Saved as {final_name}")

            for f in files:
                if f != best_model_path:
                    os.remove(f)
                    print(f"Deleted {f}")
            # Delete original best_model_path if different from final_name
            if os.path.abspath(best_model_path) != os.path.abspath(final_name):
                os.remove(best_model_path)


if __name__ == "__main__":
    consolidate()
