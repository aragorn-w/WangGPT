from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    """Configuration dataclass for the WangGPT model.

    Attributes:
    - d_vocab: Size of the vocabulary.
    - d_model: Dimension of the embedding vectors.
    - n_layers: Number of transformer blocks.
    - n_heads: Number of attention heads.
    - window_size: Maximum sequence length.
    - d_mlp: Dimension of the MLP hidden layer. Defaults to 4 * d_model.
    - dropout: Dropout probability.
    - tie_embeddings: Whether to tie token and unembedding weights.
    """
    d_vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    window_size: int
    d_mlp: Optional[int] = None
    dropout: float = 0.1
    tie_embeddings: bool = True
    use_pos_emb: bool = True

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
    - d_model: Dimension of the input features.
    - eps: Small constant for numerical stability.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization to the input tensor.

        Args:
        - x: Input tensor.

        Returns:
        - Normalized tensor.
        """
        norm: torch.Tensor = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer with causal masking.

    Args:
    - config: Configuration object.
    """
    def __init__(self, config: Config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads: int = config.n_heads
        self.d_head: int = config.d_model // config.n_heads

        # Split projections (Extra Credit: "efficient split version")
        self.q_proj: nn.Linear = nn.Linear(config.d_model, config.d_model,
                                           bias=False)
        self.k_proj: nn.Linear = nn.Linear(config.d_model, config.d_model,
                                           bias=False)
        self.v_proj: nn.Linear = nn.Linear(config.d_model, config.d_model,
                                           bias=False)
        self.o_proj: nn.Linear = nn.Linear(config.d_model, config.d_model,
                                           bias=False)

        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.window_size,
                                                           config.window_size))
                             .view(1, 1, config.window_size,
                                   config.window_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes multi-head attention.

        Args:
        - x: Input tensor of shape (B, T, C).

        Returns:
        - Attention output of shape (B, T, C).
        """
        B, T, C = x.size()

        q: torch.Tensor = self.q_proj(x).view(B, T, self.n_heads, self.d_head)\
            .transpose(1, 2)
        k: torch.Tensor = self.k_proj(x).view(B, T, self.n_heads, self.d_head)\
            .transpose(1, 2)
        v: torch.Tensor = self.v_proj(x).view(B, T, self.n_heads, self.d_head)\
            .transpose(1, 2)

        att: torch.Tensor = (q @ k.transpose(-2, -1)) * \
            (1.0 / (self.d_head ** 0.5))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y: torch.Tensor = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MLP(nn.Module):
    """Feed-forward multi-layer perceptron.

    Args:
    - config: Configuration object.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(config.d_model, config.d_mlp,
                                        bias=False)
        self.fc2: nn.Linear = nn.Linear(config.d_mlp, config.d_model,
                                        bias=False)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.act: nn.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
        - x: Input tensor.

        Returns:
        - Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block consisting of attention and MLP.

    Args:
    - config: Configuration object.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.norm1: RMSNorm = RMSNorm(config.d_model)
        self.attn: MultiHeadAttention = MultiHeadAttention(config)
        self.norm2: RMSNorm = RMSNorm(config.d_model)
        self.mlp: MLP = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer block.

        Args:
        - x: Input tensor.

        Returns:
        - Output tensor after residual connections.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WangGPT(nn.Module):
    """The full WangGPT autoregressive transformer model.

    Args:
    - config: Configuration object.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config: Config = config

        self.token_emb: nn.Embedding = nn.Embedding(config.d_vocab,
                                                    config.d_model)
        self.pos_emb: nn.Parameter = nn.Parameter(
            torch.zeros(1, config.window_size, config.d_model))
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

        self.blocks: nn.ModuleList = nn.ModuleList([Block(config) for _
                                                    in range(config.n_layers)])
        self.norm_f: RMSNorm = RMSNorm(config.d_model)
        self.lm_head: nn.Linear = nn.Linear(config.d_model, config.d_vocab,
                                            bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes model weights.

        Args:
        - module: PyTorch module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> \
    tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Performs the forward pass of the model.

        Args:
        - idx: Tensor of token indices.
        - targets: Optional ground truth token indices for loss calculation.

        Returns:
        - A tuple (logits, loss). Loss is None if targets is not provided.
        """
        _, T = idx.size()

        token_embeddings: torch.Tensor = self.token_emb(idx)
        x: torch.Tensor = token_embeddings
        if self.config.use_pos_emb:
            position_embeddings: torch.Tensor = self.pos_emb[:, :idx.size(1), :]
            x = x + position_embeddings
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits: torch.Tensor = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """Generates text autoregressively with varied sampling methods.

        Supports Temperature, Top-K, and Top-P (Nucleus) sampling.

        Args:
        - idx: Initial token indices.
        - max_new_tokens: Number of tokens to generate.
        - temperature: Sampling temperature.
        - top_k: Only sample from the top K tokens.
        - top_p: Only sample from tokens with cumulative probability < top_p.

        Returns:
        - Tensor containing the generated token indices.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop index if it exceeds window size
            idx_cond = idx if idx.size(1) <= self.config.window_size \
            else idx[:, -self.config.window_size:]

            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature

            # Optional: Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # Optional: Top-P (Nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits,
                                                           descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits,
                                                          dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token
                # above the threshold
                sorted_indices_to_remove[..., 1:] = \
                    sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for b in range(logits.size(0)):
                    indices_to_remove = \
                        sorted_indices[b, sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
