#!/usr/bin/env python3
"""
Shakespeare Word Embeddings — end to end Python demo
----------------------------------------------------
This script downloads Shakespeare (Project Gutenberg #100),
cleans & tokenizes it, trains a Word2Vec model, and provides:
  • Similar-word queries
  • Word analogies (king - man + woman ≈ queen)
  • Odd-one-out
  • A 2D t-SNE plot of selected words

Run:
  python shakespeare_embeddings.py --neighbors king queen love romeo
  python shakespeare_embeddings.py --analogy king man woman
  python shakespeare_embeddings.py --oddone romeo hamlet banana mercutio
  python shakespeare_embeddings.py --plot 500

Dependencies (auto-installed if missing):
  gensim, tqdm, scikit-learn, matplotlib, numpy, requests

All artifacts are saved next to the script:
  - data/shakespeare.txt
  - models/shakespeare.w2v (gensim format)
  - outputs/embedding_tsne.png
  - outputs/embeddings.tsv & outputs/metadata.tsv (TensorBoard Projector)
"""
from __future__ import annotations
import argparse
import collections
import io
import os
import random
import re
import sys
import time
from pathlib import Path

# ---------------------------
# Lightweight dependency bootstrap
# ---------------------------
REQS = ["gensim", "tqdm", "scikit-learn", "matplotlib", "numpy", "requests"]

def ensure_deps():
    import importlib
    missing = []
    for m in REQS:
        try:
            importlib.import_module(m.replace('-', '_'))
        except Exception:
            missing.append(m)
    if missing:
        print(f"[setup] Installing missing packages: {missing}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

ensure_deps()

# Now safely import
from tqdm import tqdm
import numpy as np
import requests
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Paths
# ---------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
for p in (DATA, MODELS, OUT):
    p.mkdir(parents=True, exist_ok=True)

RAW_PATH = DATA / "shakespeare.txt"
MODEL_PATH = MODELS / "shakespeare.w2v"
TSNE_PNG = OUT / "embedding_tsne.png"
TSV_EMB = OUT / "embeddings.tsv"
TSV_META = OUT / "metadata.tsv"

# ---------------------------
# Download + Clean
# ---------------------------
GUTENBERG_URLS = [
    # Common mirrors for ebook #100
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
    "https://www.gutenberg.org/files/100/100-0.txt",
    "https://www.gutenberg.org/ebooks/100.txt.utf-8",
]

HEADER_RE = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
FOOTER_RE = re.compile(r"\*\*\*\s*END OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)


def download_shakespeare(path: Path = RAW_PATH) -> str:
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"[download] Using cached file: {path}")
        return path.read_text(encoding="utf-8", errors="ignore")

    last_err = None
    for url in GUTENBERG_URLS:
        try:
            print(f"[download] Fetching: {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            text = r.text
            path.write_text(text, encoding="utf-8")
            print(f"[download] Saved to {path} ({len(text):,} chars)")
            return text
        except Exception as e:
            print(f"[download] Failed {url}: {e}")
            last_err = e
    raise SystemExit(f"Could not download Shakespeare corpus. Last error: {last_err}")


def strip_gutenberg_boilerplate(text: str) -> str:
    # Remove header/footer if present
    start = HEADER_RE.search(text)
    end = FOOTER_RE.search(text)
    if start and end:
        core = text[start.end():end.start()]
    else:
        core = text

    # Normalize whitespace
    core = core.replace('\r', '\n')
    core = re.sub(r"\n{3,}", "\n\n", core)
    return core

# ---------------------------
# Tokenization helpers
# ---------------------------
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def to_sentences(text: str) -> list[list[str]]:
    # Split to sentences, then use gensim.simple_preprocess for robust tokenization
    sents = []
    for s in SENT_SPLIT.split(text):
        s = s.strip()
        if not s:
            continue
        # Keep 'thy', 'thou', etc.; lower=True; deacc removes accents/punct
        tokens = simple_preprocess(s, deacc=True, min_len=2, max_len=20)
        if tokens:
            sents.append(tokens)
    return sents

# ---------------------------
# Frequency utils
# ---------------------------

def word_frequencies(sentences: list[list[str]]) -> collections.Counter:
    cnt = collections.Counter()
    for s in sentences:
        cnt.update(s)
    return cnt

# ---------------------------
# Train Word2Vec
# ---------------------------

def train_w2v(
    sentences: list[list[str]],
    vector_size: int = 200,
    window: int = 5,
    min_count: int = 3,
    workers: int = max(1, os.cpu_count() or 1),
    epochs: int = 10,
) -> Word2Vec:
    print("[w2v] Training Word2Vec…")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # skip-gram for rare Shakespearean words
        negative=10,
        sample=1e-4,
        seed=SEED,
        epochs=epochs,
    )
    model.init_sims(replace=True)  # memory trim (legacy no-op in recent gensim)
    model.save(str(MODEL_PATH))
    print(f"[w2v] Model saved to {MODEL_PATH}")
    return model

# ---------------------------
# Visualize (t‑SNE)
# ---------------------------

def tsne_plot(model: Word2Vec, words: list[str], png_path: Path = TSNE_PNG):
    vocab = [w for w in words if w in model.wv]
    if len(vocab) < 2:
        print("[plot] Not enough words in vocabulary to plot.")
        return
    X = np.vstack([model.wv[w] for w in vocab])
    print("[plot] Running t-SNE (this can take ~30–60s)…")
    tsne = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=min(30, max(5, len(vocab)//3)))
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X2[:, 0], X2[:, 1])
    for (x, y), w in zip(X2, vocab):
        plt.annotate(w, xy=(x, y), xytext=(5, 2), textcoords="offset points")
    plt.title("Word Embeddings (t‑SNE)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print(f"[plot] Saved: {png_path}")

    # Also export TSV for TensorBoard projector
    np.savetxt(TSV_EMB, X, delimiter="\t")
    with open(TSV_META, "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")
    print(f"[proj] Projector TSVs saved: {TSV_EMB.name}, {TSV_META.name}")

# ---------------------------
# Interactive helpers
# ---------------------------

def neighbors(model: Word2Vec, words: list[str], topn: int = 10):
    for w in words:
        if w not in model.wv:
            print(f"[neighbors] '{w}' not in vocab — try a different word.")
            continue
        sims = model.wv.most_similar(w, topn=topn)
        print(f"\nTop {topn} neighbors of '{w}':")
        for i, (tok, score) in enumerate(sims, 1):
            print(f"  {i:2d}. {tok:>15s}   cos={score:.3f}")


def analogy(model: Word2Vec, a: str, b: str, c: str, topn: int = 10):
    for w in (a, b, c):
        if w not in model.wv:
            print(f"[analogy] '{w}' not in vocab.")
            return
    print(f"\nAnalogy: {a} - {b} + {c} ≈ ?")
    res = model.wv.most_similar(positive=[a, c], negative=[b], topn=topn)
    for i, (tok, score) in enumerate(res, 1):
        print(f"  {i:2d}. {tok:>15s}   cos={score:.3f}")


def odd_one_out(model: Word2Vec, words: list[str]):
    vocab = [w for w in words if w in model.wv]
    missing = [w for w in words if w not in model.wv]
    if missing:
        print(f"[oddone] Missing from vocab (ignored): {missing}")
    if len(vocab) < 3:
        print("[oddone] Need at least 3 known words.")
        return
    odd = model.wv.doesnt_match(vocab)
    print(f"Odd one out of {vocab}: {odd}")

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train and explore Word2Vec on Shakespeare (Gutenberg #100)")
    ap.add_argument("--force-retrain", action="store_true", help="Ignore cached model and retrain")
    ap.add_argument("--vector-size", type=int, default=200)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--neighbors", nargs="*", default=[], help="Words to list nearest neighbors for")
    ap.add_argument("--analogy", nargs=3, metavar=("A", "B", "C"), help="Solve analogy A - B + C ≈ ?")
    ap.add_argument("--oddone", nargs="+", help="Find the odd word out")
    ap.add_argument("--plot", type=int, default=0, help="Plot t‑SNE of top N frequent words (0 disables)")
    args = ap.parse_args()

    # Load or train
    if MODEL_PATH.exists() and not args.force_retrain:
        print(f"[load] Loading cached model: {MODEL_PATH}")
        model = Word2Vec.load(str(MODEL_PATH))
    else:
        raw = download_shakespeare()
        raw = strip_gutenberg_boilerplate(raw)
        print("[prep] Tokenizing…")
        sentences = to_sentences(raw)
        print(f"[prep] Prepared {len(sentences):,} sentences; tokens≈{sum(map(len, sentences)):,}")
        model = train_w2v(
            sentences,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            epochs=args.epochs,
        )

    # Quick demo defaults if nothing specified
    if not (args.neighbors or args.analogy or args.oddone or args.plot):
        args.neighbors = ["king", "queen", "love", "romeo", "witch", "money"]
        args.analogy = ("king", "man", "woman")
        args.oddone = ["romeo", "hamlet", "banana", "mercutio"]
        args.plot = 400

    # Frequencies for plotting selection
    if args.plot:
        print("[freq] Computing word frequencies for plot selection…")
        # If we trained in this session, we have 'sentences'; otherwise, recompute quickly
        if 'sentences' not in locals():
            raw = RAW_PATH.read_text(encoding="utf-8", errors="ignore")
            raw = strip_gutenberg_boilerplate(raw)
            sentences = to_sentences(raw)
        freq = word_frequencies(sentences)
        # pick top-N frequent words that are in vocab and alphabetic
        chosen = []
        for w, _ in freq.most_common():
            if w in model.wv and re.fullmatch(r"[a-z]+", w):
                chosen.append(w)
            if len(chosen) >= args.plot:
                break
        tsne_plot(model, chosen)

    # Interactions
    if args.neighbors:
        neighbors(model, args.neighbors)
    if args.analogy:
        analogy(model, *args.analogy)
    if args.oddone:
        odd_one_out(model, args.oddone)

    print("\n[done] Enjoy exploring Shakespearean word vectors! ✨")


if __name__ == "__main__":
    main()
