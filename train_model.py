import gc
import os
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from utils import clean_tokenization,\
    UNIFIED_VOCAB, TOK_TO_IDX, VOCAB_SIZE
from wang_gpt import WangGPT


import matplotlib.pyplot as plt

MAX_NUM_PARAMS = 10_000_000

def get_batch(data: torch.Tensor, iter: int, batch_size: int,
              win_size: int, device: str):
    """Generates a batch of training inputs and targets.

    Slices the tokenized data into windows of size 'win_size' and shifts
    them by one token to create input-target pairs for next-token prediction.

    Args:
    - data: The full tensor of token indices.
    - iter: The current iteration number for determining start position.
    - batch_size: Number of training windows per batch.
    - win_size: The number of tokens per training window.
    - device: The torch device (cuda/cpu) to place the tensors on.

    Returns:
    - A tuple (x, y) where x is the input batch and y is the target batch.
    """

    max_idx = len(data) - win_size - 1
    if max_idx <= 0:
        return \
            torch.zeros((batch_size, win_size), dtype=torch.long).to(device), \
                torch.zeros((batch_size, win_size), dtype=torch.long).to(device)
    start_pos = (iter * batch_size) % max_idx
    indices = [(start_pos + i) % max_idx for i in range(batch_size)]
    x = torch.stack([data[idx: idx + win_size] for idx in indices])
    y = torch.stack([data[idx + 1: idx + 1 + win_size] for idx in indices])
    return x.to(device), y.to(device)


def get_optimal_dm(vocab_size, num_tbs, window_size, use_pe, target_params=MAX_NUM_PARAMS):
    """Calculates the optimal embedding dimension d_m to hit a parameter target.

    Uses a quadratic solver to find the d_m that brings the total parameter
    count closest to 'target_params', ensuring h=4 attention head compatibility.

    Args:
    - vocab_size: Size of the vocabulary.
    - num_tbs: Number of transformer blocks.
    - window_size: Size of the context window.
    - use_pe: Boolean indicating if positional encodings are used.
    - target_params: The desired total parameter count (default 10M).

    Returns:
    - The integer d_m (clamped to multiple of 4).
    """

    a = 12 * num_tbs
    b = 2 * vocab_size + (window_size if use_pe else 0) + 2 * num_tbs + 1
    c = -target_params

    # Quadratic formula
    dm = (-b + (b**2 - 4*a*c)**0.5) / (2*a)

    # Round to nearest multiple of 4 (for 4 attention heads)
    dm = int(round(dm / 4) * 4)
    return dm


def train_on_file(file_path, config, all_tokens, curve_id=""):
    """Executes a single training run for a specific model configuration.

    Handles model initialization, DataParallel distribution, AdamW optimization,
    and logging to wandb and stdout.

    Args:
    - file_path: Path to the data file.
    - config: Dictionary containing hyperparameters (lr, bs, layers, etc).
    - all_tokens: List of token strings for the entire dataset.
    - curve_id: Identifier string for grouping curves in logging.

    Returns:
    - A tuple (last_loss, loss_history, state_dict, final_config, vocab).
    """

    import utils
    vocab_size = utils.VOCAB_SIZE
    tok_to_idx = utils.TOK_TO_IDX
    unified_vocab = utils.UNIFIED_VOCAB

    # Extract file name for logging
    file_name = os.path.basename(file_path).split(".")[0]

    # Dynamically calculate d_m
    dm = get_optimal_dm(vocab_size, config["layers"], config["window_size"],
                        config["use_pe"])
    config["d_m"] = dm

    run_name = f"{curve_id}-lr{config["lr"]}-bs{config["batch_size"]}-dm{dm}"
    print(f"\n>>> STARTING RUN: {run_name} (V: {vocab_size}, d_m: {dm}, "
          "PE: {config[\"use_pe\"]})")

    split_idx = int(len(all_tokens) * 0.8)
    train_tokens = all_tokens[:split_idx]
    train_data = torch.tensor([tok_to_idx[w] for w in train_tokens],
                              dtype=torch.long)

    # Initialize Weights and Biases logging
    wandb.init(
        project="star-wars-gpt",
        name=run_name,
        group=curve_id,
        config={**config, "file": file_name, "vocab_size": vocab_size},
        reinit=True
    )

    # Set up the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WangGPT(
        d_v=vocab_size,
        d_m=config["d_m"],
        num_tbs=config["layers"],
        w=config["window_size"],
        b=config["batch_size"],
        use_pe=config["use_pe"]
    )

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model_for_params = model.module if isinstance(model, nn.DataParallel) \
        else model
    total_params = sum(p.numel() for p in model_for_params.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"],
                            betas=(0.9, 0.95), weight_decay=0.1)
    loss_func = nn.CrossEntropyLoss()

    loss_history = []
    model.train()
    for iter in range(config["iters"]):
        inputs, outputs = get_batch(train_data, iter, config["batch_size"],
                                    config["window_size"], device)
        logits = model(inputs)
        loss = loss_func(logits.view(-1, vocab_size), outputs.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            loss_val = loss.item()
            loss_history.append((iter, loss_val))
            wandb.log({"loss": loss_val}, step=iter)
            if iter % 5000 == 0:
                print(f"Iter {iter:<5} | Loss: {loss_val:.4f}")

    wandb.finish()

    # Return everything needed for best-model selection
    return loss_val, loss_history, model_for_params.state_dict(), config, \
        unified_vocab


def save_plot(curves, title, filename):
    """Generates a loss curve comparison plot using Matplotlib.

    Args:
    - curves: Dict with model labels as keys, values as (iter, loss) histories.
    - title: String for the plot title.
    - filename: Output file path for the plot.
    """

    plt.figure(figsize=(10, 6))
    for name, history in curves.items():
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
    # Ensure unified vocab exists
    from unify_vocabs import main as build_vocab
    build_vocab()

    # Re-import vocab after building to get updated values
    import utils
    import importlib
    importlib.reload(utils)

    # Target data file
    file_path = "data/combined_star_wars.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    all_tokens = utils.clean_tokenization(text_data)

    lrs = [3e-4, 5e-4]
    batch_sizes = [32, 64]
    base_config = {
        "layers": 7,
        "window_size": 128,
        "iters": 50_000
    }

    best_curves = {}

    # Random model (no training) baseline
    print("\n--- Generating Random Initialization Model ---")
    dm_rand = get_optimal_dm(utils.VOCAB_SIZE, base_config["layers"],
                             base_config["window_size"], True)
    rand_model = WangGPT(utils.VOCAB_SIZE, dm_rand, base_config["layers"],
                         base_config["window_size"], 32, use_pe=True)

    # Calculate random loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rand_model.to(device)
    split_idx = int(len(all_tokens) * 0.8)
    train_tokens = all_tokens[:split_idx]
    train_data = torch.tensor([utils.TOK_TO_IDX[w] for w in train_tokens],
                              dtype=torch.long)
    inputs, outputs = get_batch(train_data, 0, 32, 128, device)
    with torch.no_grad():
        logits = rand_model(inputs)
        rand_loss = nn.CrossEntropyLoss()(logits.view(-1, utils.VOCAB_SIZE),
                                          outputs.view(-1)).item()

    # Create a "curve" of constant random loss
    best_curves["random_init"] = [(0, rand_loss), (base_config["iters"], rand_loss)]
    torch.save({
        "model_state_dict": rand_model.state_dict(),
        "config": {**base_config, "d_m": dm_rand, "use_pe": True},
        "vocab": utils.UNIFIED_VOCAB
    }, "models/random_init.pt")
    del rand_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train no position-encodings models
    print("\n--- Starting No-PE Grid Search ---")
    best_no_pe_loss = float("inf")
    best_no_pe_state = None
    best_no_pe_config = None
    best_no_pe_vocab = None

    for lr, bs in product(lrs, batch_sizes):
        config = {**base_config, "lr": lr, "batch_size": bs, "use_pe": False}
        last_loss, history, state, cfg, vocab = train_on_file(file_path,
                                                              config,
                                                              all_tokens, "no_pe")
        if last_loss < best_no_pe_loss:
            best_no_pe_loss = last_loss
            best_curves["no_pe_best"] = history
            best_no_pe_state = state
            best_no_pe_config = cfg
            best_no_pe_vocab = vocab

    torch.save({"model_state_dict": best_no_pe_state,
                "config": best_no_pe_config, "vocab": best_no_pe_vocab},
                "models/no_pe_best.pt")

    # Train full (has positional encodings) models
    print("\n--- Starting With-PE Grid Search ---")
    best_with_pe_loss = float("inf")
    best_with_pe_state = None
    best_with_pe_config = None
    best_with_pe_vocab = None

    for lr, bs in product(lrs, batch_sizes):
        config = {**base_config, "lr": lr, "batch_size": bs, "use_pe": True}
        last_loss, history, state, cfg, vocab = train_on_file(file_path, config,
                                                              all_tokens,
                                                              "with_pe")
        if last_loss < best_with_pe_loss:
            best_with_pe_loss = last_loss
            best_curves["with_pe_best"] = history
            best_with_pe_state = state
            best_with_pe_config = cfg
            best_with_pe_vocab = vocab

    torch.save({"model_state_dict": best_with_pe_state,
                "config": best_with_pe_config, "vocab": best_with_pe_vocab},
                "models/with_pe_best.pt")

    # Save final comparison plot
    save_plot(best_curves, "Star Wars GPT Model Comparison",
              "training_comparison.png")
    print("\n>>> ALL TRAINING COMPLETE. Comparison plot saved to "
          "training_comparison.png.")
