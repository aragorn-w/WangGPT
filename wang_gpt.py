import torch
import torch.nn as nn
import torch.nn.functional as F

# shape notes to self are row-major (e.g. (# of rows, # of cols))
# and 1-based for dimension index


class EmbeddingLookup(nn.Module):
    def __init__(self, d_v: int, d_m: int):
        """Initializes the embedding lookup table.

        Args:
        - d_v: Vocabulary size (number of unique tokens).
        - d_m: Embedding dimension size.
        """
        super().__init__()
        # shape = (d_m, d_v)
        self.lookup = nn.Parameter(torch.randn(d_v, d_m) * 0.02)

    def forward(self, token_indices: list[int]):
        """Converts token indices into their corresponding embedding vectors.

        Args:
        - token_indices: A list or tensor of token indices.

        Returns:
        - A tensor of embeddings with shape (batch_size, sequence_length, d_m).
        """
        if isinstance(token_indices, list):
            token_indices = torch.tensor(token_indices, dtype=torch.long,
                                         device=self.lookup.device)
        if token_indices.dim() == 1:
            token_indices = token_indices.unsqueeze(0)
        # shape = (b, t, d_m)
        return self.lookup[token_indices]

class RMSNorm(nn.Module):
    def __init__(self, d_m: int):
        """Initializes the Root Mean Square Layer Normalization.

        Args:
        - d_m: Dimension of the input features to normalize.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_m))
        self.epsilon = 1e-6

    def forward(self, X: torch.tensor):
        """Applies RMS normalization to the input tensor.

        Args:
        - X: Input tensor of shape (batch_size, sequence_length, d_m).

        Returns:
        - Normalized tensor of the same shape.
        """
        # X has dimensions (b, t, d_m)
        recip_rms: torch.tensor = torch.rsqrt(X.pow(2).mean(-1, keepdim=True)
                                          + self.epsilon)
        X = (X * recip_rms) * self.gamma
        return X

class MultiHeadTransformerBlock(nn.Module):
    def __init__(self, d_m: int, h: int):
        """Initializes a Multi-Head Attention Transformer Block.

        Args:
        - d_m: Embedding dimension size.
        - h: Number of attention heads.
        """
        super().__init__()
        # h = # of attention heads
        self.h: int = h
        self.d_k: int = d_m // h

        # Combine the query, key, and value weight matrices into one
        # to linearly project to the Q, K, and V matrices in parallel
        # shape = (d_m, 3*d_m)
        self.W_QKV: torch.tensor = nn.Parameter(torch.randn(d_m, 3*d_m) * 0.02)
        # shape = (d_m, d_m)
        self.W_O: torch.tensor = nn.Parameter(torch.randn(d_m, d_m) * 0.02)
        # The number of hidden neurons for the MLP feed forward layer
        self.d_ff: int = 4 * d_m
        self.Wff_in = nn.Parameter(torch.randn(d_m, self.d_ff) * 0.02)
        self.Wff_out = nn.Parameter(torch.randn(self.d_ff, d_m) * 0.02)

        # RMS normalization layers
        self.rms_norm_1 = RMSNorm(d_m)
        self.rms_norm_2 = RMSNorm(d_m)

    def forward(self, X: torch.tensor):
        """Performs the forward pass of the transformer block.

        Args:
        - X: Input tensor of shape (batch_size, sequence_length, d_m).

        Returns:
        - Contextualized tensor of the same shape.
        """
        b: int
        t: int
        d_m: int
        b, t, d_m = X.shape

        # shape = (b, t, d_m) x (d_m, 3*d_m) = (b, t, 3*d_m)
        QKV: torch.Tensor = torch.matmul(self.rms_norm_1(X), self.W_QKV)
        Q: torch.Tensor
        K: torch.Tensor
        V: torch.Tensor
        # RMS normalize X, compute the QKV concatenated tensor, then
        # split it into chunks of d_m size along token index (3rd dim)
        # all Q, K, V shapes = (b, t, d_m)
        Q, K, V = QKV.split(d_m, dim=2)
        # all Q, V, K shapes = (b, t, h, d_k)^T(2, 3) = (b, h, t, d_k)
        Q = Q.view(b, t, self.h, self.d_k).transpose(1, 2)
        K = K.view(b, t, self.h, self.d_k).transpose(1, 2)
        V = V.view(b, t, self.h, self.d_k).transpose(1, 2)

        # Calculate the attention scores for each directional word-to-word
        # (use tensor contraction of non-batch dimensions)
        # shape = (b, h, t, d_k) x (b, h, d_k, t)
        # treat as (t, d_k) x (d_k, t) = (t, t)
        # and as batch, (b, h, t, t)
        A_scores: torch.tensor = torch.matmul(Q, K.transpose(-2, -1))
        # Trick from original transformer paper to prevent gradient explosion
        A_scores /= self.d_k**0.5

        # Hide future tokens from each head
        M: torch.tensor = torch.triu(
            torch.ones(b, self.h, t, t, device=X.device), diagonal=1)
        A_scores = A_scores.masked_fill(M == 1, float("-inf"))

        # Normalize self-attention scores of non-future tokens
        A_scores = F.softmax(A_scores, dim=-1)

        # Contextualize the attention scores and prepare to merge the heads
        # (use tensor contraction of non-batch dimensions)
        # shape = ((b, h, t, t) x (b, h, t, d_k))^T(2, 3)
        # treat as ((t, t) x (t, d_k))^T(2, 3) = (t, d_k)^T(2, 3)
        # and as batch, (b, h, t, d_k)^T(2, 3) = (b, t, h, d_k)
        A_values: torch.tensor = torch.matmul(A_scores, V).transpose(1, 2)
        # Merge the heads
        # shape = (b, t, h, d_k) -> (b, t, h*d_k) -> (b, t, d_m)
        A_values = A_values.contiguous().view(b, t, d_m)
        # Tensor contraction of non-batch dimensions and "mixing" of heads
        # shape = (b, t, d_m) x (d_m, d_m)
        # treat as (t, d_m) x (d_m, d_m) = (t, d_m)
        # and as batch, (b, t, d_m)
        A_values = torch.matmul(A_values, self.W_O)
        # Add projected attention values back to input embeddings as residual
        X = X + A_values

        # MLP layer to sparsely emphasize most important embedding dims
        # Pass to MLP feedforward layer's hidden neurons
        # (use tensor contraction of non-batch dimensions)
        # shape = (b, t, d_m) x (d_m, d_ff)
        # treat as (t, d_m) x (d_m, d_ff) = (t, d_ff)
        # and as batch, (b, t, d_ff)
        FF_values: torch.tensor = F.relu(torch.matmul(self.rms_norm_2(X),
                                                      self.Wff_in))
        # Go from MLP FF layer's hidden neurons to output neurons
        # (use tensor contraction of non-batch dimensions)
        # shape = (b, t, d_ff) x (d_ff, d_m)
        # treat as (t, d_ff) x (d_ff, d_m) = (t, d_m)
        # and as batch, (b, t, d_m)
        FF_values = torch.matmul(FF_values, self.Wff_out)
        # Add MLP feed forward outputs back to X as second residual
        X = X + FF_values

        return X


class WangGPT(nn.Module):
    def __init__(self, d_v: int, d_m: int, num_tbs: int, w: int, b: int, use_pe: bool = True):
        """Initializes the Star Wars DIY-GPT model.

        Args:
        - d_v: Vocabulary size.
        - d_m: Embedding dimension size.
        - num_tbs: Number of transformer blocks to stack.
        - w: Window size (max sequence length).
        - b: Batch size (for internal tracking).
        - use_pe: Whether to apply positional encodings.
        """
        super().__init__()
        self.d_v = d_v # Number of vocabulary words
        self.d_m = d_m # Number of embedding dimensions
        self.num_tbs = num_tbs # Number of transformer blocks
        self.w = w # Number of tokens in training window
        self.b = b # Number of training windows per batch
        self.use_pe = use_pe # Toggle for positional encodings
        self.E = EmbeddingLookup(d_v, d_m)
        # PE = positional encoding
        if use_pe:
            self.PE = nn.Parameter(torch.randn(w, d_m) * 0.02)
        self.TBs = nn.ModuleList([MultiHeadTransformerBlock(d_m, 4)
                                  for _ in range(num_tbs)])
        self.rms_norm = RMSNorm(d_m)
        self.U = nn.Parameter(torch.randn(d_m, d_v) * 0.01)

    def forward(self, tok_idxs):
        """Performs the forward pass to predict the next tokens.

        Args:
        - tok_idxs: Tensor of token indices with shape (batch_size, sequence_length).

        Returns:
        - Logits for each vocabulary word with shape (batch_size, sequence_length, d_v).
        """
        # Convert token indices to embedding vectors
        X = self.E(tok_idxs)

        # Add token positional encodings
        if self.use_pe:
            X = X + self.PE[:X.shape[1], :]

        # Self-attention layer to contextualize window of embeddings
        for TB in self.TBs:
            X = TB(X)

        # RMS normalize the output before unembedding it for the classifier
        X = self.rms_norm(X)

        # Unembedding matrix to derive predicted token indices
        # (use tensor contraction of non-batch dimensions)
        # shape = (b, t, d_m) x (d_m, d_v) = (b, t, d_v)
        X = torch.matmul(X, self.U)
        return X
