import os
import re

import torch


class Tokenizer:
    """Word-level tokenizer that splits on non-alphabetic characters.

    Lowercases the input, extracts alphabetic runs, and maps them to integer
    indices via a vocabulary built from a training corpus.
    """

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.tok_to_idx: dict[str, int] = {tok: idx for idx, tok in enumerate(vocab)}
        self.idx_to_tok: dict[int, str] = {idx: tok for tok, idx in self.tok_to_idx.items()}
        self.vocab_size: int = len(vocab)

    def tokenize(self, text: str) -> list[str]:
        """Split text into lowercase alphabetic tokens."""
        return re.findall(r"[a-z]+", text.lower())

    def encode(self, text: str) -> list[int]:
        """Convert text to a list of token indices, skipping unknown tokens."""
        return [self.tok_to_idx[t] for t in self.tokenize(text) if t in self.tok_to_idx]

    def decode(self, ids: list[int]) -> str:
        """Convert token indices back to a space-joined string."""
        return " ".join(self.idx_to_tok[i] for i in ids if i in self.idx_to_tok)

    def save(self, path: str):
        """Save the vocabulary list to a .pt file."""
        torch.save(self.vocab, path)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load a tokenizer from a saved vocabulary file."""
        vocab = torch.load(path, weights_only=True)
        return cls(vocab)

    @classmethod
    def from_corpus(cls, text: str) -> "Tokenizer":
        """Build a tokenizer by extracting all unique tokens from a corpus."""
        tokens = re.findall(r"[a-z]+", text.lower())
        vocab = sorted(set(tokens))
        return cls(vocab)


# Module-level convenience: load unified vocab if available
try:
    _TOKENIZER = Tokenizer.load("models/unified_vocab.pt")
    TOK_TO_IDX = _TOKENIZER.tok_to_idx
    IDX_TO_TOK = _TOKENIZER.idx_to_tok
    VOCAB_SIZE = _TOKENIZER.vocab_size
except Exception:
    _TOKENIZER = None
    TOK_TO_IDX = None
    IDX_TO_TOK = None
    VOCAB_SIZE = 0


def clean_tokenization(text: str) -> list[str]:
    """Tokenize text into lowercase alphabetic words.

    Kept for backward compatibility with training scripts.
    """
    return re.findall(r"[a-z]+", text.lower())


def translate_state_dict(
    old_sd: dict[str, torch.Tensor], d_m: int
) -> dict[str, torch.Tensor]:
    """Maps legacy model weight keys to the current architecture's keys."""
    new_sd = {}
    for k, v in old_sd.items():
        if k == "E.lookup":
            new_sd["token_emb.weight"] = v
            new_sd["lm_head.weight"] = v  # Tied
        elif k == "PE":
            new_sd["pos_emb"] = v.unsqueeze(0)
        elif "TBs." in k:
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
