# WangGPT: A Transformer from Scratch

*Completed March 3, 2026 for MATH 498C.*

WangGPT is an autoregressive Transformer-based Large Language Model (LLM)
implemented from scratch in PyTorch. This project explores the architecture of modern LLMs, including multi-head attention, RMS normalization, and advanced sampling techniques.

## File Map

| File | Description |
| :--- | :--- |
| `wang_gpt.py` | Core model: `Config` dataclass, `RMSNorm`, `MultiHeadAttention`, `MLP`, `Block`, `WangGPT` class with `forward()`, `generate()`, and `generate_text()` |
| `utils.py` | Custom word-level `Tokenizer` class with `encode()`/`decode()`, vocabulary loading, legacy state dict translation |
| `train_model.py` | Full training pipeline: batching, parameter budget optimizer, grid search, W&B logging, checkpointing. Supports `--quick` for fast test runs |
| `chat.py` | CLI text generation: loads a checkpoint and generates from `--prompt` or `--prompts-file` |
| `unify_vocabs.py` | Data preprocessing: unzips corpus, consolidates source texts, builds unified vocabulary |
| `consolidate_models.py` | Utility to evaluate and prune saved checkpoints |
| `count_tokens.py` | Counts tokens per source file for corpus statistics |

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### 0. Download the Data
Star Wars novelization data files are [hosted here on my Google Drive](https://drive.google.com/file/d/1y1fF0ZwVFuTmOiQM8XHDNkcBIaDDbym5/view?usp=share_link).

**Reproduction**: Simply move the `data.zip` file into the project root. The training script will automatically unzip and process it.

### 0.1 **Weights & Biases (Optional)**: To skip logging to a cloud account, disable W&B before training:
```bash
export WANDB_MODE=disabled
```

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Training

Full training (~10k iterations, takes a while):
```bash
uv run train_model.py
```

**Quick test run** (~200 iterations on a small corpus subset, finishes in under 5 minutes):
```bash
uv run train_model.py --quick
```

### 3. Generate Text

Generate from a prompt:
```bash
uv run chat.py --model models/with_pe_best.pt --prompt "luke skywalker"
```

Generate from a file of prompts (one per line):
```bash
uv run chat.py --model models/with_pe_best.pt --prompts-file prompts.txt
```

Options: `--max-tokens`, `--temperature`, `--top-k`, `--top-p` (see `uv run chat.py --help`).

---

## Architecture & Design Choices

### Core Features
- **Config Dataclass**: All hyperparameters are managed via a `Config` dataclass (no hardcoding).
- **Multi-Head Attention (MHA)**: Implemented with $h=4$ heads.
- **Efficient Split Projections**: Separate `nn.Linear` layers for $W_Q, W_K, W_V$, and $W_O$, with a reduced head dimension $d_{head} = d_{model} / h$.
- **Custom Word Tokenizer**: Built from scratch with `encode(text) -> list[int]` and `decode(ids) -> str` methods. Lowercases and splits on non-alphabetic characters.
- **Advanced Sampling**: The `.generate()` method supports **Temperature**, **Top-K**, and **Top-P (Nucleus)** sampling. The `.generate_text()` method provides a text-in/text-out interface.
- **Tied Embeddings**: The input embedding weights are tied with the output unembedding layer to reduce parameters and improve training efficiency.
- **Custom Training Loop**: Implemented from scratch (no HuggingFace/Lightning) with full **Batching** support.

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
| **Random Init** | Completed | 10.58 |
| **No-PE** | Completed | 7.12 |
| **With-PE** | Completed | 7.09 |

### Example Generations

| Model | Sample Output (Prompt: "LUKE:") |
| :--- | :--- |
| **Untrained** | `luke hool stammered toweled clinical shorting toolooking nella optimists diners stressing...` |
| **Trained (No-PE)** | `luke point heard watch later talk glad breath battle away better ve exactly watch going happened matter empire let changed choice deep slow breath believe welcome moment...` |
| **Trained (With-PE)** | `luke empire yoda force going talk time believe nt needed exactly matter empire... (converging)` |

*Note: The trained models successfully learned Star Wars vocabulary (Empire, Yoda, Luke) and basic sentence structures despite the small 10M parameter size.*

---

## Retrospective

### Dataset
The model was trained on a concatenated corpus of Star Wars novelizations. The dataset was pre-processed into an ASCII-only format with a unified vocabulary of ~37,000 tokens.

### Design Decisions
- Implementing **RMSNorm** and **AdamW** from the start led to much faster convergence than standard LayerNorm. Using **separate projections** for heads made the code more readable and easier to debug.

### Technical Challenges
- Synchronizing gradients across a mixed GPU setup (4080s, 3090s, 4060 Ti) introduced synchronization overhead. This was resolved using `nn.DataParallel` and adjusting batch sizes to fit the smallest VRAM card (4060 Ti).

### Key Takeaways
- Tokenization approach matters a lot. The initial version delegated to spacy which was slow (~1hr for the full corpus) and obscured what the tokenizer was actually doing. Replacing it with a simple regex-based word tokenizer made the pipeline much faster and more transparent.

### Future Directions
If I were to do this over, I would implement **Byte Pair Encoding (BPE)** to handle sub-words better and reduce the vocabulary size, which currently consumes a significant portion of the parameter budget.

---

## Contributions
Developed by Aragorn Wang. Covers the full stack: model architecture (`wang_gpt.py`), custom tokenizer (`utils.py`), training loop (`train_model.py`), and CLI generation interface (`chat.py`).
