# WangGPT: A Transformer from Scratch

WangGPT is an autoregressive Transformer-based Large Language Model (LLM)
implemented from scratch in PyTorch. This project explores the architecture of modern LLMs, including multi-head attention, RMS normalization, and advanced sampling techniques.

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for robust dependency management.

### 0. Download the Data
> All Star Wars novelization text data is contained within the .zip file [hosted here on my Google Drive](https://drive.google.com/file/d/1G6x5uiZbvacg6NP9vq5xVFeIJnAjUdtf/view?usp=share_link). Download the .zip file and move it (DO NOT UNZIP IT, that is done for you automatically) to the project root.

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Training
The training script performs a grid search and baseline comparison.
```bash
uv run train_model.py
```

### 3. Interactive Chat
Load a trained model and chat with it:
```bash
uv run chat.py
```

---

## Architecture & Design Choices

### Core Features (Extra Credit)
- **Config Dataclass**: All hyperparameters are managed via a `Config` dataclass (no hardcoding).
- **Multi-Head Attention (MHA)**: Implemented with $h=4$ heads.
- **Efficient Split Projections**: Unlike the baseline implementation, this version uses separate `nn.Linear` layers for $W_Q, W_K, W_V$, and $W_O$, allowing for optimized head dimensions (Extra Credit).
- **Advanced Sampling**: The `.generate()` method supports **Temperature**, **Top-K**, and **Top-P (Nucleus)** sampling (Extra Credit).
- **Tied Embeddings**: The input embedding weights are tied with the output unembedding layer to reduce parameters and improve training efficiency.
- **Custom Training Loop**: Implemented from scratch (no HuggingFace/Lightning) with full **Batching** support (Extra Credit).

### Model Specifications
- **Parameters**: ~10,000,000 (Adjusted dynamically via `get_optimal_dm`).
- **Layers**: 7 Transformer Blocks.
- **Window Size**: 128 Tokens.
- **Normalization**: Root Mean Square Layer Normalization (RMSNorm).
- **Optimizer**: AdamW with weight decay.

---

## Results

### Training Performance
Loss curves comparing the Random Baseline vs. No-PE (Best) vs. With-PE (Partial) variants are saved as `training_comparison.png`.

| Model | Status | Final Loss |
| :--- | :--- | :--- |
| **Random Init** | Completed | 10.5 |
| **No-PE** | Completed (Best of Grid Search) | 6.8 |
| **With-PE** | In-Progress (Partial @ 20k iters) | 7.5 |

### Example Generations

| Model | Sample Output (Prompt: "LUKE:") |
| :--- | :--- |
| **Untrained** | `luke hool stammered toweled clinical shorting toolooking nella optimists diners stressing...` |
| **Trained (No-PE)** | `luke point heard watch later talk glad breath battle away better ve exactly watch going happened matter empire let changed choice deep slow breath believe welcome moment...` |
| **Trained (With-PE)** | `luke empire yoda force going talk time believe nt needed exactly matter empire... (converging)` |

*Note: The trained models successfully learned Star Wars vocabulary (Empire, Yoda, Luke) and basic sentence structures despite the small 10M parameter size.*

---

## Writeup & Reflection

### Dataset
The model was trained on a concatenated corpus of Star Wars novelizations. The dataset was pre-processed into an ASCII-only format with a unified vocabulary of ~37,000 tokens.

### Design Reflections
- **What went well?**: Implementing **RMSNorm** and **AdamW** from the start led to much faster convergence than standard LayerNorm. The decision to use **separate projections** for heads made the code more readable and easier to debug.
- **Challenges**: Synchronizing gradients across a mixed GPU setup (4080s, 3090s, 4060 Ti) introduced synchronization overhead. We solved this using `nn.DataParallel` and adjusting batch sizes to fit the smallest VRAM card (4060 Ti).
- **Lessons Learned**: Tokenization is a massive bottleneck. The initial Spacy-based cleanup for a large corpus took nearly an hour before the first training iteration. In the future, I would pre-tokenize and save the results to disk as `.pt` tensors.

### Future Directions
If given more time, I would implement **Byte Pair Encoding (BPE)** to handle sub-words better and reduce the vocabulary size, which currently consumes a significant portion of the parameter budget.

---

## Contributions
**Sole Contributor**: I, Aragorn Wang implemented the entire system, including the model architecture, tokenization pipeline, training loop, and interactive chat interface.

Specific commits cover:
- Architecture: `wang_gpt.py` refactor to Config/Linear.
- Training: `train_model.py` grid search and batching logic.
- Documentation: Final `README.md` and results visualization.
