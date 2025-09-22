import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
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
def train_bpe(all_texts, model_prefix="bpe", vocab_size=8000):
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
# 3) Encode sentences -> token IDs
# ----------------------------
def encode_sentence(sentence, sp, add_bos=False, add_eos=False):
    ids = sp.encode(sentence, out_type=int)
    if add_bos:
        ids = [sp.bos_id()] + ids
    if add_eos:
        ids = ids + [sp.eos_id()]
    return ids

# ----------------------------
# 4) Dataset + DataLoader (ID tensors)
# ----------------------------
class ParallelTextDataset(Dataset):
    def __init__(self, modern_sentences, shake_sentences, sp):
        assert len(modern_sentences) == len(shake_sentences), "Parallel corpora must be aligned!"
        self.modern = modern_sentences
        self.shake = shake_sentences
        self.sp = sp

    def __len__(self):
        return len(self.modern)

    def __getitem__(self, idx):
        # Encoder input: no BOS; Decoder input: BOS + target; Target: target + EOS
        src_ids = encode_sentence(self.modern[idx], self.sp, add_bos=False, add_eos=True)
        tgt_in_ids = encode_sentence(self.shake[idx], self.sp, add_bos=True, add_eos=False)
        tgt_out_ids = encode_sentence(self.shake[idx], self.sp, add_bos=False, add_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_in_ids, dtype=torch.long), torch.tensor(tgt_out_ids, dtype=torch.long)

def pad_sequence(seqs, pad_id):
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :s.size(0)] = s
    return out

def collate_fn(batch):
    # batch: list of (src_ids, tgt_in_ids, tgt_out_ids)
    pad_id = 1
    src_list, tgt_in_list, tgt_out_list = zip(*batch)
    src = pad_sequence(src_list, pad_id)
    tgt_in = pad_sequence(tgt_in_list, pad_id)
    tgt_out = pad_sequence(tgt_out_list, pad_id)
    return src, tgt_in, tgt_out

# ----------------------------
# 5) Seq2Seq Model (token IDs)
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_id, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_ids):
        # src_ids: [B, Ls]
        emb = self.dropout(self.embedding(src_ids))        # [B, Ls, E]
        outputs, hidden = self.rnn(emb)                    # outputs: [B, Ls, 2H], hidden: [2*L, B, H]
        # merge bi-directional features
        H = outputs.size(2) // 2
        outputs = outputs[:, :, :H] + outputs[:, :, H:]    # [B, Ls, H]
        # merge last-layer fwd/bwd hidden to one layer
        L2, B, Hh = hidden.size()                          # L2=2*num_layers
        hidden = hidden[0:L2:2] + hidden[1:L2:2]           # [num_layers, B, H]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: [B, H]; enc_outputs: [B, Ls, H]
        B, Ls, H = enc_outputs.shape
        dec = dec_hidden.unsqueeze(1).repeat(1, Ls, 1)     # [B, Ls, H]
        energy = torch.tanh(self.attn(torch.cat([dec, enc_outputs], dim=2)))  # [B, Ls, H]
        scores = self.v(energy).squeeze(2)                 # [B, Ls]
        attn = F.softmax(scores, dim=1)                    # [B, Ls]
        return attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_id, attention, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attention = attention
        self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, enc_outputs):
        # input_token: [B] next decoder token id
        emb = self.dropout(self.embedding(input_token)).unsqueeze(1)  # [B, 1, E]
        # attention over encoder outputs
        dec_hidden_last = hidden[-1]                                   # [B, H]
        attn_weights = self.attention(dec_hidden_last, enc_outputs)    # [B, Ls]
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)    # [B, 1, H]
        rnn_in = torch.cat([emb, context], dim=2)                      # [B, 1, E+H]
        output, hidden = self.rnn(rnn_in, hidden)                      # output: [B,1,H]
        logits = self.fc_out(torch.cat([output, context], dim=2)).squeeze(1)  # [B, V]
        return logits, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_id, bos_id, eos_id):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def forward(self, src_ids, tgt_in_ids, tgt_out_ids, teacher_forcing_ratio=0.5):
        """
        src_ids:    [B, Ls]
        tgt_in_ids: [B, Lt]   (input side, with BOS at start)
        tgt_out_ids:[B, Lt]   (output side, with EOS at end)
        """
        B, Lt = tgt_out_ids.shape
        enc_outputs, hidden = self.encoder(src_ids)

        logits_seq = []
        inp = tgt_in_ids[:, 0]  # first BOS token

        for t in range(Lt):
            logits, hidden, _ = self.decoder(inp, hidden, enc_outputs)  # [B, V]
            logits_seq.append(logits.unsqueeze(1))
            teacher = (torch.rand(1).item() < teacher_forcing_ratio)
            if t + 1 < Lt:
                inp = tgt_in_ids[:, t+1] if teacher else logits.argmax(dim=1)

        return torch.cat(logits_seq, dim=1)  # [B, Lt, V]

    @torch.no_grad()
    def translate(self, src_ids, max_len=60):
        """
        Greedy decoding until EOS.
        src_ids: [B, Ls]
        """
        self.eval()
        enc_outputs, hidden = self.encoder(src_ids)
        B = src_ids.size(0)
        inp = torch.full((B,), self.bos_id, dtype=torch.long, device=src_ids.device)

        outputs = []
        for _ in range(max_len):
            logits, hidden, _ = self.decoder(inp, hidden, enc_outputs)
            next_ids = logits.argmax(dim=1)  # greedy next token
            outputs.append(next_ids)
            inp = next_ids
            if torch.all(next_ids == self.eos_id):
                break

        if len(outputs) == 0:
            return torch.empty((B, 0), dtype=torch.long, device=src_ids.device)
        return torch.stack(outputs, dim=1)  # [B, T]




# ----------------------------
# 6) Training Loop
# ----------------------------
def train_model(model, dataloader, sp, epochs=5, lr=1e-3, device=None):
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pad_id = sp.pad_id()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    model.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for src, tgt_in, tgt_out in dataloader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt_in, tgt_out, teacher_forcing_ratio=0.5)  # [B, Lt, V]

            B, Lt, V = logits.shape
            loss = criterion(logits.reshape(B*Lt, V), tgt_out.reshape(B*Lt))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    print("Training complete.")


# ----------------------------
# 7) Inference Helper
# ----------------------------
@torch.no_grad()
def translate_sentence(model, sentence, sp, device=None, max_len=60):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    # encode src with EOS
    src_ids = encode_sentence(sentence, sp, add_bos=False, add_eos=True)
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    pred_ids = model.translate(src, max_len=max_len)  # [1, T]
    # strip everything after EOS
    ids = pred_ids.squeeze(0).tolist()
    if sp.eos_id() in ids:
        ids = ids[:ids.index(sp.eos_id())]
    # decode subwords
    return sp.decode(ids)

# ----------------------------
# 8) Main
# ----------------------------
def main(modern_dir="data/modern", shake_dir="data/shakespeare",
         batch_size=32, epochs=5, retrain=False, vocab_size=8000, emb_dim=256, hidden_dim=256):

    modern_texts = read_text_files(modern_dir)
    shake_texts  = read_text_files(shake_dir)
    all_texts = modern_texts + shake_texts
    if not all_texts:
        raise RuntimeError("No text files found in modern/ or shakespeare/")

    sp = train_bpe(all_texts, vocab_size=vocab_size)
    pad_id, bos_id, eos_id = sp.pad_id(), sp.bos_id(), sp.eos_id()
    vocab_size = sp.get_piece_size()

    dataset = ParallelTextDataset(modern_texts, shake_texts, sp)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    enc = Encoder(vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, pad_id=pad_id)
    attn = Attention(hidden_dim)
    dec = Decoder(vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, pad_id=pad_id, attention=attn)
    model = Seq2Seq(enc, dec, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)

    ckpt = "shakespeare_model.pth"
    if (not retrain) and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        print(f"Loaded pretrained model from {ckpt}")
    else:
        print("Starting training...")
        train_model(model, dataloader, sp, epochs=epochs)
        torch.save(model.state_dict(), ckpt)
        print(f"Model saved to {ckpt}")

    # Demo inference
    example_sentence = "Where are you going?"
    pred_text = translate_sentence(model, example_sentence, sp)
    print("Original:", example_sentence)
    print("Predicted Shakespearean:", pred_text)

    return sp, dataloader, model

# ----------------------------
# 9) CLI
# ----------------------------
if __name__ == "__main__":
    sp, dataloader, model = main()
