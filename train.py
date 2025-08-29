# %%
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

# device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print(
    "Device name:",
    (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else "No GPU found"
    ),
)
# %%
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary Size: {vocab_size}")
print("".join(chars))
# %%
# character level encoder/decoder
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}


def encode(text):
    return [stoi[c] for c in text]


def decode(indices):
    return "".join([itos[i] for i in indices])


print(encode("Hello World! I bring peace!"))
print(decode([random.randint(0, vocab_size - 1) for _ in range(42)]))
# %%
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])
# %%
split_percent = 0.9
n = int(data.shape[0] * split_percent)

train_data = data[:n]
val_data = data[n:]
# %%
block_size = 8
train_data[:block_size]
# %%
# understanding example of how blocksize is used

x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(
        f"when input is {context} ({decode(context.tolist())}) the target: {target} ({itos[int(target)]})"
    )
# %%
torch.manual_seed(1337)  # seed set to match tutorial

batch_size = 4  # independent sequences in parallel
block_size = 8  # maximum context length for predictions


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print(f"inputs shape: {xb.shape}")
print(xb)
print(f"targets shape: {yb.shape} ")
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(
            f"when input is {context} ({decode(context.tolist())}) the target: {target} ({itos[int(target)]})"
        )
    print("=" * 42)


# %%
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
# %%
decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist())
# %%
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# %%
batch_size = 32
for steps in range(10000):
    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
# %%
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
# %%
