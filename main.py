# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import spacy

# %%
class EmbeddingLookup(nn.Module):
    def __init__(self, d_v: int, d_m: int):
        super().__init__()
        self.lookup = nn.Parameter(torch.randn(d_m, d_v) * 0.02)

    def forward(self, token_indices: list[int]):
        X = self.lookup[:, token_indices]
        return X

class TransformerBlock(nn.Module):
    def __init__(self, d_m: int):
        super().__init__()
        self.d_m = d_m
        self.W_Q = nn.Parameter(torch.randn(d_m, d_m) * 0.02)
        self.W_K = nn.Parameter(torch.randn(d_m, d_m) * 0.02)
        self.W_O = nn.Parameter(torch.randn(d_m, d_m) * 0.02)
        self.W_V = nn.Parameter(torch.randn(d_m, d_m) * 0.02)
        self.mlp_in = nn.Parameter(torch.randn(d_m, d_m) * 0.02)
        self.mlp_out = nn.Parameter(torch.randn(d_m, d_m) * 0.02)

    def forward(self, X):
        X = X.T
        num_tok = X.shape[0]
        Q = torch.matmul(X, self.W_Q)
        K = torch.matmul(X, self.W_K)
        A_scores = torch.matmul(Q, K.T)

        M = torch.triu(torch.ones(num_tok, num_tok), diagonal=1)
        A_scores = A_scores.masked_fill(M == 1, float("-inf"))

        A_scores = F.softmax(A_scores, dim=-1)

        RH = torch.matmul(X, self.W_V)
        RH = torch.matmul(RH, self.W_O)
        A_scores = torch.matmul(A_scores, RH)

        X = X + A_scores

        mlp_result = F.relu(torch.matmul(X, self.mlp_in))
        mlp_result = torch.matmul(mlp_result, self.mlp_out)

        X = X + mlp_result

        return X.T


class GPT(nn.Module):
    def __init__(self, d_v, d_m, num_tbs):
        super().__init__()
        self.d_v = d_v
        self.d_m = d_m
        self.num_tbs = num_tbs
        self.E = EmbeddingLookup(d_v, d_m)
        self.TBs = nn.ModuleList([TransformerBlock(d_m) for _ in range(num_tbs)])
        self.U = nn.Parameter(torch.randn(d_m, d_v) * 0.01)

    def forward(self, tok_idxs):
        X = self.E(tok_idxs)
        for TB in self.TBs:
            X = TB(X)
        X = torch.matmul(X.T, self.U)
        return X

# %%
NUM_TRAINING_ITERS = 1000
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
BETA_WEIGHTS = (0.9, 0.95)

if __name__ == "__main__":
    # Text cleaning and tokenization utilities
    nlp_model = spacy.load("en_core_web_sm")
    def clean_tokenization(text):
        lower_text = nlp_model(text.lower())
        cleaned_tokens = [
            token.lemma_ for token in lower_text
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        return cleaned_tokens

    # Clean and tokenize the test set text
    training_text = None
    with open("training_text.txt", "r", encoding="utf-8") as file:
        training_text = file.read()
    assert training_text is not None

    train_tokens = clean_tokenization(training_text)
    vocab = set(train_tokens)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # Clean and tokenize the test set text
    testing_text = None
    with open("testing_text.txt", "r", encoding="utf-8") as file:
        testing_text = file.read()
    test_tokens = clean_tokenization(testing_text)

    # Initialize the GPT, loss function, and training optimizer
    d_v = len(vocab)
    wang_gpt = GPT(d_v, d_m=300, num_tbs=3)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        wang_gpt.parameters(),
        lr=LEARNING_RATE,
        betas=BETA_WEIGHTS,
        weight_decay=WEIGHT_DECAY
    )

    # Train the single-head self-attention decoder GPT
    for iter in range(NUM_TRAINING_ITERS):
        pred_probs = wang_gpt(blahblahinputs)
        loss = loss_func(pred_probs, blahblahtargets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print(f"Iter {iter}, Loss: {loss.item():.4f}")
    print("===TRAINING COMPLETE===")
