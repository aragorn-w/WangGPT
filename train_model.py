import importlib
import os
from itertools import product
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from utils import clean_tokenization,\
    UNIFIED_VOCAB, TOK_TO_IDX, VOCAB_SIZE, translate_state_dict
from wang_gpt import WangGPT, Config
from unify_vocabs import main as build_vocab

MAX_NUM_PARAMS = 10_000_000


def get_batch(data: torch.Tensor, iter: int, batch_size: int,
              win_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a batch of input-target pairs from the training data.

    Args:
    - data: Tensor containing all tokenized training data.
    - iter: Current iteration number for batch offset calculation.
    - batch_size: Number of sequences per batch.
    - win_size: Length of each sequence.
    - device: Device to move the tensors to (e.g., 'cuda' or 'cpu').

    Returns:
    - A tuple (x, y) where x is the input batch and y is the target batch.
    """
    max_idx: int = len(data) - win_size - 1
    if max_idx <= 0:
        return \
            torch.zeros((batch_size, win_size), dtype=torch.long).to(device), \
                torch.zeros((batch_size, win_size), dtype=torch.long).to(device)
    start_pos: int = (iter * batch_size) % max_idx
    indices: list[int] = [(start_pos + i) % max_idx for i in range(batch_size)]
    x: torch.Tensor = torch.stack([data[idx: idx + win_size] for idx
                                   in indices])
    y: torch.Tensor = torch.stack([data[idx + 1: idx + 1 + win_size] for
                                   idx in indices])
    return x.to(device), y.to(device)

def get_optimal_dm(vocab_size: int, num_tbs: int, window_size: int, target_params: int = MAX_NUM_PARAMS) -> int:
    """Calculates the optimal d_model to fit within a parameter budget.

    Uses a quadratic formula derived from the model's architecture.

    Args:
    - vocab_size: Size of the vocabulary.
    - num_tbs: Number of transformer blocks.
    - window_size: Maximum sequence length.
    - target_params: Maximum allowed parameters.

    Returns:
    - The optimal embedding dimension (d_m) as an integer multiple of 4.
    """
    # Updated formula for the new architecture:
    # token_emb: V * dm
    # pos_emb: window * dm
    # blocks (L): 4 * dm^2 (attention) + 2 * dm * (4 * dm) (mlp) = 12 * dm^2
    # norm_f: dm
    # lm_head: dm * V (tied with token_emb, so we count V*dm once)
    # Total ~ L * 12 * dm^2 + (V + window + 1) * dm
    a: int = 12 * num_tbs
    b: int = vocab_size + window_size + 1
    c: int = -target_params
    dm: float = (-b + (b**2 - 4*a*c)**0.5) / (2*a)
    dm_int: int = int(round(dm / 4) * 4)
    return dm_int

def train_on_file(config_dict: dict[str, Any], all_tokens: list[str], curve_id: str = "") -> tuple[float, list[tuple[int, float]], dict[str, torch.Tensor], dict[str, Any], Optional[list[str]]]:
    """Trains a model on a specific text file with given hyperparameters.

    Handles dataset splitting, logging to Weights & Biases, and checkpointing.

    Args:
    - config_dict: Dictionary containing hyperparameters (lr, layers, etc.).
    - all_tokens: List of pre-tokenized words from the corpus.
    - curve_id: Label for identifying this run in plots.

    Returns:
    - A tuple (last_loss, loss_history, model_state_dict, config, vocab).
    """
    import utils
    vocab_size = utils.VOCAB_SIZE
    tok_to_idx = utils.TOK_TO_IDX

    dm = get_optimal_dm(vocab_size, config_dict["layers"],
                        config_dict["window_size"])
    config_dict["d_m"] = dm
    run_name = \
        f"{curve_id}-lr{config_dict['lr']}-bs{config_dict['batch_size']}-dm{dm}"
    checkpoint_path = f"models/{run_name}.pt"

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Check if it's truly finished or just a periodic checkpoint
        if len(ckpt.get("history", [])) >= (config_dict["iters"] // 100):
            print(f">>> SKIPPING: {run_name} (Found completed run)")
            return ckpt["last_loss"], ckpt["history"], \
                ckpt["model_state_dict"], ckpt["config"], ckpt["vocab"]
        else:
            print(f">>> RESUMING: {run_name} from iteration "
                  f"{ckpt['history'][-1][0]}")
            start_iter = ckpt["history"][-1][0]
            loss_history = ckpt["history"]
    else:
        print(f"\n>>> STARTING RUN: {run_name} (V: {vocab_size}, d_m: {dm})")
        start_iter = 0
        loss_history = []

    split_idx = int(len(all_tokens) * 0.8)
    train_tokens = all_tokens[:split_idx]
    train_data = torch.tensor([tok_to_idx[w] for w in train_tokens
                               if w in tok_to_idx], dtype=torch.long)

    wandb.init(
        project="star-wars-gpt",
        name=run_name,
        group=curve_id,
        config={**config_dict, "vocab_size": vocab_size},
        reinit=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = Config(
        d_vocab=vocab_size,
        d_model=dm,
        n_layers=config_dict["layers"],
        n_heads=4,
        window_size=config_dict["window_size"]
    )
    model = WangGPT(model_config)

    if start_iter > 0:
        model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=config_dict["lr"],
                            betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    last_loss = 0
    for iter in range(start_iter, config_dict["iters"]):
        inputs, outputs = get_batch(train_data, iter,
                                    config_dict["batch_size"],
                                    config_dict["window_size"], device)
        _, loss = model(inputs, targets=outputs)

        # Average loss if using DataParallel
        if isinstance(model, nn.DataParallel):
            loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            last_loss = loss.item()
            loss_history.append((iter, last_loss))
            wandb.log({"loss": last_loss}, step=iter)
            if iter % 5000 == 0:
                print(f"Iter {iter:<5} | Loss: {last_loss:.4f}")
                # Periodic checkpoint
                state_dict = model.module.state_dict() if isinstance(
                    model, nn.DataParallel) else model.state_dict()
                torch.save({
                    "last_loss": last_loss,
                    "history": loss_history,
                    "model_state_dict": state_dict,
                    "config": config_dict,
                    "vocab": utils.UNIFIED_VOCAB
                }, checkpoint_path)

    wandb.finish()

    state_dict = model.module.state_dict() if isinstance(
        model, nn.DataParallel) else model.state_dict()

    torch.save({
        "last_loss": last_loss,
        "history": loss_history,
        "model_state_dict": state_dict,
        "config": config_dict,
        "vocab": utils.UNIFIED_VOCAB
    }, checkpoint_path)

    return last_loss, loss_history, state_dict, config_dict, utils.UNIFIED_VOCAB

def save_plot(curves: dict, title: str, filename: str):
    """Generates and saves a Matplotlib plot of training loss curves.

    Args:
    - curves: Dictionary mapping model names to their loss history lists.
    - title: Title of the plot.
    - filename: Path to save the resulting image file.
    """
    plt.figure(figsize=(10, 6))
    for name, history in curves.items():
        if not history: continue
        iters, losses = zip(*history)
        plt.plot(iters, losses, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Build initial vocab if it doesn't exist
    build_vocab()

    # Reload utils so VOCAB_SIZE and other constants are updated
    import utils
    importlib.reload(utils)

    file_path = "data/combined_star_wars.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    all_tokens = utils.clean_tokenization(text_data)

    lrs = [3e-4, 5e-4]
    batch_sizes = [32, 64]
    base_config = {"layers": 7, "window_size": 128, "iters": 50_000}
    best_curves = {}

    # Establish random baseline
    if os.path.exists("models/random_init.pt"):
        rand_ckpt = torch.load("models/random_init.pt", map_location="cpu")
        best_curves["random_init"] = rand_ckpt.get("history", [])
    else:
        print("\n--- Generating Random Initialization Model ---")
        dm_rand = get_optimal_dm(utils.VOCAB_SIZE, base_config["layers"],
                                 base_config["window_size"])
        model_config = Config(utils.VOCAB_SIZE, dm_rand, base_config["layers"],
                              4, base_config["window_size"])
        rand_model = WangGPT(model_config)

        torch.save({
            "model_state_dict": rand_model.state_dict(),
            "config": {**base_config, "d_m": dm_rand, "use_pe": True},
            "vocab": utils.UNIFIED_VOCAB,
            "history": [(0, 10.0), (base_config["iters"], 10.0)] # Placeholder
        }, "models/random_init.pt")
        best_curves["random_init"] = [(0, 10.0), (50000, 10.0)]

    # Grid search
    for use_pe in [True, False]:
        curve_id = "with_pe" if use_pe else "no_pe"
        best_file = f"models/{curve_id}_best.pt"

        print(f"\n--- Starting {curve_id} Grid Search ---")
        best_loss = float("inf")
        best_data = None

        for lr, bs in product(lrs, batch_sizes):
            config = {**base_config, "lr": lr, "batch_size": bs,
                      "use_pe": use_pe}
            res = train_on_file(config, all_tokens, curve_id)
            last_loss, history, state, cfg, vocab = res

            if last_loss < best_loss:
                best_loss = last_loss
                best_data = {"model_state_dict": state, "config": cfg,
                             "vocab": vocab, "history": history}

        if best_data:
            torch.save(best_data, best_file)
            best_curves[f"{curve_id}_best"] = best_data["history"]

    save_plot(best_curves, "Star Wars GPT Model Comparison",
              "training_comparison.png")
    print("\n>>> ALL TRAINING COMPLETE.")
