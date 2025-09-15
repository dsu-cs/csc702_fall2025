import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import random

# -------------------------------
# 1. Load and clean the texts
# -------------------------------
with open("csc702_fall2025\words_to_emb\Lucas\juliet.txt", "r", encoding="utf-8") as f:
    romeo = f.read()
with open("csc702_fall2025\words_to_emb\Lucas\moby.txt", "r", encoding="utf-8") as f:
    moby = f.read()

text = romeo + "\n" + moby
text = re.sub(r"[^a-zA-Z0-9\s,.?!;:']", " ", text).lower()
words = text.split()

# -------------------------------
# 2. Build vocabulary
# -------------------------------
vocab = sorted(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
vocab_size = len(vocab)

print("Vocab size:", vocab_size)

# -------------------------------
# 3. Dataset for word sequences
# -------------------------------
class WordDataset(Dataset):
    def __init__(self, words, seq_len=10):
        self.words = words
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.words) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.words[idx:idx+self.seq_len]
        y = self.words[idx+1:idx+self.seq_len+1]
        return torch.tensor([word_to_idx[w] for w in x]), torch.tensor([word_to_idx[w] for w in y])

seq_len = 15
dataset = WordDataset(words, seq_len)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# -------------------------------
# 4. Define LSTM model
# -------------------------------
class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

model = LM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# -------------------------------
# 5. Train briefly
# -------------------------------
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.transpose(1,2), y)  # shape fix
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, loss: {total_loss/len(loader):.4f}")

# -------------------------------
# 6. Text generation
# -------------------------------
def generate(model, start_text="love is", max_len=30):
    model.eval()
    words_out = start_text.split()
    input_ids = torch.tensor([[word_to_idx.get(w, 0) for w in words_out]])
    hidden = None

    for _ in range(max_len):
        logits, hidden = model(input_ids[:, -seq_len:], hidden)
        probs = torch.softmax(logits[:, -1], dim=-1).squeeze()
        next_id = torch.multinomial(probs, 1).item()
        words_out.append(idx_to_word[next_id])
        input_ids = torch.tensor([[next_id]])
    
    return " ".join(words_out)

# Example blended generations
print("\nGenerated samples:")
for seed in ["juliet whispered", "the whale spoke", "love and sea"]:
    print(">", generate(model, seed, max_len=20))
