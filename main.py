# %%
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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
        # Compute naive self-attention scores
        X = X.T
        num_tok = X.shape[0]
        Q = torch.matmul(X, self.W_Q)
        K = torch.matmul(X, self.W_K)
        A_scores = torch.matmul(Q, K.T)

        # Hide future tokens
        M = torch.triu(torch.ones(num_tok, num_tok), diagonal=1)
        A_scores = A_scores.masked_fill(M == 1, float("-inf"))

        # Normalize self-attention scores of non-future tokens
        A_scores = F.softmax(A_scores, dim=-1)

        # Contextualize X with relevant embedding dims
        RH = torch.matmul(X, self.W_V)
        RH = torch.matmul(RH, self.W_O)
        A_values = torch.matmul(A_scores, RH)
        X = X + A_values

        # MLP layer to sparsely emphasize most important embedding dims
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
        # Convert token indices to embedding vectors
        X = self.E(tok_idxs)

        # Self-attention layer to contextualize window of embeddings
        for TB in self.TBs:
            X = TB(X)

        # Unembedding matrix to derive predicted token indices
        X = torch.matmul(X.T, self.U)
        return X

# %%
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
train_data = torch.tensor([word_to_idx[w] for w in train_tokens], dtype=torch.long)

# Clean and tokenize the test set text
testing_text = None
with open("testing_text.txt", "r", encoding="utf-8") as file:
    testing_text = file.read()
assert testing_text is not None
test_tokens = clean_tokenization(testing_text)

# %%
# Set training hyperparameters
NUM_TRAINING_ITERS = 5_000
ITER_PRINT_MULTIPLE = 1_000
ITER_LOGGING_MULTIPLE = 100
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
BETA_WEIGHTS = (0.9, 0.95)
TRAIN_WINDOW_SIZE = 128

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

# Print the number of parameters, ensure <10M
pass

# Train the single-head self-attention decoder GPT
last_win_start_idx = len(train_tokens) - TRAIN_WINDOW_SIZE - 1
loss_history = {}
for iter in range(NUM_TRAINING_ITERS):
    win_start_idx = torch.randint(0, last_win_start_idx, size=(1,)).item()
    inputs = train_data[win_start_idx: win_start_idx + TRAIN_WINDOW_SIZE]
    outputs = train_data[win_start_idx + 1: win_start_idx + 1 + TRAIN_WINDOW_SIZE]
    pred_probs = wang_gpt(inputs)
    loss = loss_func(pred_probs.view(-1, d_v), outputs.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_scalar = loss.item()
    if iter % ITER_PRINT_MULTIPLE == 0:
        print(f"Iter {iter:<8} - Loss: {loss_scalar:.4f}")
    if iter % ITER_LOGGING_MULTIPLE == 0:
        loss_history[iter] = loss_scalar
print("===TRAINING COMPLETE===")

# Graph loss function
plt.figure(figsize=(10, 5))
plt.plot(list(loss_history.keys()), list(loss_history.values()))
plt.xlabel("Training Iteration")
plt.ylabel("Cross-entropy Loss")
plt.title("WangGPT Training Loss History")
plt.show()

# %%
# Example inference of trained GPT
EXAMPLE_INPUT_PROMPT = ""
pass