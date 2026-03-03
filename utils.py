import hashlib
import os

import spacy
import torch

# Load Spacy once, disable unnecessary components
NLP_MODEL = spacy.load("en_core_web_sm", disable=["parser", "ner"])
NLP_MODEL.max_length = 2_000_000_000

# Load unified vocabulary if it exists
try:
    UNIFIED_VOCAB = torch.load("models/unified_vocab.pt")
    TOK_TO_IDX = {tok: idx for idx, tok in enumerate(UNIFIED_VOCAB)}
    IDX_TO_TOK = {idx: tok for tok, idx in TOK_TO_IDX.items()}
    VOCAB_SIZE = len(UNIFIED_VOCAB)
except Exception as _:
    UNIFIED_VOCAB = None
    TOK_TO_IDX = None
    IDX_TO_TOK = None
    VOCAB_SIZE = 0


def clean_tokenization(text: str) -> list[str]:
    """Cleans and tokenizes the input text using Spacy with disk caching.

    Lowercase the text, then removes stop words, punctuation, and
    non-alphabetic tokens to ensure only meaningful words remain.

    Args:
    - text: The raw input string to tokenize.

    Returns:
    - A list of cleaned token strings.
    """
    # Create cache directory if it doesn't exist
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Generate hash of input text to identify cache file
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"tokens_{text_hash}.pt")

    if os.path.exists(cache_path):
        print(f">>> Loading tokens from cache: {cache_path}")
        return torch.load(cache_path)

    print(">>> Tokenizing data (this may take a while)...")
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    doc = NLP_MODEL(text.lower())
    tokens = [token.text for token in doc if not token.is_space and token.is_alpha]

    # Save to cache
    torch.save(tokens, cache_path)
    return tokens


def translate_state_dict(
    old_sd: dict[str, torch.Tensor], d_m: int
) -> dict[str, torch.Tensor]:
    """Maps legacy model weight keys to the new architecture's keys.

    Args:
    - old_sd: State dict from an older version of the model.
    - d_m: Embedding dimension of the old model.

    Returns:
    - A state dict compatible with the current WangGPT class.
    """
    new_sd = {}
    for k, v in old_sd.items():
        if k == "E.lookup":
            new_sd["token_emb.weight"] = v
            new_sd["lm_head.weight"] = v  # Tied
        elif k == "PE":
            new_sd["pos_emb"] = v.unsqueeze(0)
        elif "TBs." in k:
            # TBs.0.W_QKV -> blocks.0.attn.q_proj.weight, etc.
            block_idx = k.split(".")[1]
            suffix = ".".join(k.split(".")[2:])
            if suffix == "W_QKV":
                q, k_proj, v_proj = v.split(d_m, dim=1)
                new_sd[f"blocks.{block_idx}.attn.q_proj.weight"] = q.t()
                new_sd[f"blocks.{block_idx}.attn.k_proj.weight"] = k_proj.t()
                new_sd[f"blocks.{block_idx}.attn.v_proj.weight"] = v_proj.t()
            elif suffix == "W_O":
                new_sd[f"blocks.{block_idx}.attn.o_proj.weight"] = v.t()
            elif suffix == "Wff_in":
                new_sd[f"blocks.{block_idx}.mlp.fc1.weight"] = v.t()
            elif suffix == "Wff_out":
                new_sd[f"blocks.{block_idx}.mlp.fc2.weight"] = v.t()
            elif "rms_norm_1" in suffix:
                new_sd[f"blocks.{block_idx}.norm1.weight"] = v
            elif "rms_norm_2" in suffix:
                new_sd[f"blocks.{block_idx}.norm2.weight"] = v
        elif k == "rms_norm.gamma":
            new_sd["norm_f.weight"] = v
    return new_sd
