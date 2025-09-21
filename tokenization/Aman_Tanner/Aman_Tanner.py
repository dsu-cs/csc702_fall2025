import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torchtext.vocab import GloVe
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# 1) Read book files
# ----------------------------
def read_text_files(dir_path: str):
    texts = []
    if not os.path.isdir(dir_path):
        return texts
    for fname in os.listdir(dir_path):
        if fname.endswith(".txt"):
            with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
    return texts

# ----------------------------
# 2) Train SentencePiece BPE
# ----------------------------
def train_bpe(all_texts, model_prefix="bpe", vocab_size=16000):
    combined_file = f"{model_prefix}_combined.txt"
    with open(combined_file, "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(t.replace("\n", " ") + "\n")
    spm.SentencePieceTrainer.Train(
        input=combined_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        unk_id=0, pad_id=1, bos_id=2, eos_id=3
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

# ----------------------------
# 3) Load GloVe embeddings
# ----------------------------
def load_glove(dim=100):
    print("Loading pretrained GloVe embeddings...")
    vectors = GloVe(name="6B", dim=dim)
    return vectors

# ----------------------------
# 4) Convert sentence -> embeddings tensor
# ----------------------------
def sentence_to_embeddings(sentence, tokenizer, embeddings):
    tokens = tokenizer.encode(sentence, out_type=str)
    vectors = []
    for tok in tokens:
        if tok in embeddings.stoi:
            vectors.append(embeddings[tok])
    if not vectors:
        return torch.zeros((1, embeddings.dim))
    return torch.stack(vectors)

# ----------------------------
# 5) Dataset + DataLoader
# ----------------------------
class ParallelTextDataset(Dataset):
    def __init__(self, modern_sentences, shake_sentences, tokenizer, embeddings):
        assert len(modern_sentences) == len(shake_sentences), "Parallel corpora must be aligned!"
        self.modern = modern_sentences
        self.shake = shake_sentences
        self.tokenizer = tokenizer
        self.embeddings = embeddings

    def __len__(self):
        return len(self.modern)

    def __getitem__(self, idx):
        src_emb = sentence_to_embeddings(self.modern[idx], self.tokenizer, self.embeddings)
        tgt_emb = sentence_to_embeddings(self.shake[idx], self.tokenizer, self.embeddings)
        return src_emb, tgt_emb

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lengths = [s.size(0) for s in src_batch]
    tgt_lengths = [t.size(0) for t in tgt_batch]
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)

    src_padded = torch.zeros(len(batch), max_src_len, src_batch[0].size(1))
    tgt_padded = torch.zeros(len(batch), max_tgt_len, tgt_batch[0].size(1))
    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :s.size(0), :] = s
        tgt_padded[i, :t.size(0), :] = t

    return src_padded, tgt_padded

# ----------------------------
# 6) Seq2Seq Model
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        outputs, hidden = self.rnn(x)
        outputs = outputs[:, :, :hidden.size(2)//2] + outputs[:, :, hidden.size(2)//2:]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, attention):
        super().__init__()
        self.attention = attention
        self.rnn = nn.GRU(input_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, input_step, hidden, encoder_outputs):
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((input_step, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = torch.cat((output, context), dim=2)
        output = self.fc(output)
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len, _ = tgt.size()
        outputs = torch.zeros(batch_size, tgt_len, tgt.size(2), device=src.device)

        encoder_outputs, hidden = self.encoder(src)
        hidden = hidden[:1]

        input_step = tgt[:, 0, :].unsqueeze(1)
        for t in range(tgt_len):
            output, hidden, attn = self.decoder(input_step, hidden, encoder_outputs)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if t+1 < tgt_len:
                input_step = tgt[:, t+1, :].unsqueeze(1) if teacher_force else output
        return outputs

# ----------------------------
# 7) Training Loop
# ----------------------------
def train_model(model, dataloader, epochs=5, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        for src_batch, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            optimizer.zero_grad()
            output = model(src_batch, tgt_batch)
            loss = criterion(output, tgt_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    print("Training complete.")

# ----------------------------
# 8) Inference
# ----------------------------
def translate_sentence(model, sentence, tokenizer, embeddings, max_len=50, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        src = sentence_to_embeddings(sentence, tokenizer, embeddings).unsqueeze(0).to(device)
        encoder_outputs, hidden = model.encoder(src)
        hidden = hidden[:1]
        input_step = src[:, 0, :].unsqueeze(1)
        outputs = []
        for _ in range(max_len):
            output, hidden, attn = model.decoder(input_step, hidden, encoder_outputs)
            outputs.append(output.squeeze(1).cpu())
            input_step = output
        return outputs

# ----------------------------
# 9) Embeddings → Tokens → Text
# ----------------------------
def embeddings_to_tokens(pred_embeddings, glove, top_k=1):
    if isinstance(pred_embeddings, list):
        pred_embeddings = torch.stack(pred_embeddings)
    tokens = []
    pred_norm = F.normalize(pred_embeddings, dim=1)
    vocab_norm = F.normalize(glove.vectors, dim=1)
    for vec in pred_norm:
        cos_sim = torch.matmul(vocab_norm, vec)
        top_idx = torch.topk(cos_sim, top_k).indices[0].item()
        token = glove.itos[top_idx]
        tokens.append(token)
    return tokens

def tokens_to_text(tokens, tokenizer):
    return tokenizer.decode(tokens)

# ----------------------------
# 10) Main function
# ----------------------------
def main(modern_dir="data/modern", shake_dir="data/shakespeare", batch_size=16, epochs=5):
    modern_texts = read_text_files(modern_dir)
    shake_texts = read_text_files(shake_dir)
    all_texts = modern_texts + shake_texts

    if not all_texts:
        raise RuntimeError("No text files found in modern/ or shakespeare/")

    sp = train_bpe(all_texts)
    glove = load_glove()
    dataset = ParallelTextDataset(modern_texts, shake_texts, sp, glove)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    input_dim = glove.dim
    hidden_dim = 256
    enc = Encoder(input_dim, hidden_dim)
    attn = Attention(hidden_dim, hidden_dim)
    dec = Decoder(input_dim, hidden_dim, input_dim, attn)
    model = Seq2Seq(enc, dec)

    print("Starting training...")
    train_model(model, dataloader, epochs=epochs)

    # Demo inference
    example_sentence = "Where are you going?"
    output_embeddings = translate_sentence(model, example_sentence, sp, glove)
    pred_tokens = embeddings_to_tokens(output_embeddings, glove)
    pred_text = tokens_to_text(pred_tokens, sp)

    print("Original:", example_sentence)
    print("Predicted Shakespearean:", pred_text)
    return sp, glove, dataloader, model

# ----------------------------
# 11) CLI
# ----------------------------
if __name__ == "__main__":
    sp, glove, dataloader, model = main()