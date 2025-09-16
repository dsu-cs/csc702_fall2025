import collections
import io
import os
import random
import re
import sys
import time
from pathlib import Path



from tqdm import tqdm
import numpy as np
import httpx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
for p in (DATA, MODELS, OUT):
    p.mkdir(parents=True, exist_ok=True)

SEED = 0xDEADBEEF
random.seed(SEED)

RAW_PATH = DATA / "lovecraft.txt"
MODEL_PATH = MODELS / "lovecraft.w2v"
TSNE_PNG = OUT / "embedding_tsne.png"
TSV_EMB = OUT / "lovecraft_embeddings.tsv"
TSV_META = OUT / "lovecraft_metadata.tsv"


def get_lovecraft_text():
    if RAW_PATH.exists():
        print(f"[data] Loading existing text from {RAW_PATH}")
        return RAW_PATH.read_text(encoding="utf-8")
    url = "https://archive.org/stream/the-complete-works-of-h.-p.-lovecraft_202107/The%20Complete%20Works%20Of%20H.P.%20Lovecraft_djvu.txt"
    response = httpx.get(url)
    response.raise_for_status()
    fullpage =  response.text
    fullpage = fullpage.split('The Haunter of the Dark 00.0.2... 0c. ccccecccecceccecceecceccecceeececceeeceeceecaeeceececaueceeeeeaeeseeeeeseeeaees 694 ')[1]
    fullpage = "\n\n".join([x.strip() for x in fullpage.split('Return to Table of Contents ')[:-1]])
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        f.write(fullpage)
    return fullpage

# Ashar's code. -----------------------------------------------------------------------
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


def word_frequencies(sentences: list[list[str]]) -> collections.Counter:
    cnt = collections.Counter()
    for s in sentences:
        cnt.update(s)
        if "euclidean" in s:
            print(s)
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
    
    if MODEL_PATH.exists():
        print(f"[w2v] Loading existing model from {MODEL_PATH}")
        return Word2Vec.load(str(MODEL_PATH))
    
    print("[w2v] Training Word2Vec…")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        negative=10,
        sample=1e-4,
        seed=SEED,
        epochs=epochs,
    )

    model.save(str(MODEL_PATH))
    print(f"[w2v] Model saved to {MODEL_PATH}")
    return model


def neighbors(model: Word2Vec, words: list[str], topn: int = 10):
    for w in words:
        if w not in model.wv:
            print(f"[neighbors] '{w}' not in vocab — try a different word.")
            continue
        sims = model.wv.most_similar(w, topn=topn)
        print(f"\nTop {topn} neighbors of '{w}':")
        for i, (tok, score) in enumerate(sims, 1):
            print(f"  {i:2d}. {tok:>15s}   cos={score:.3f}")

# End Ashar's code. --------------------------------------------------------------------

text = get_lovecraft_text()
sentences = to_sentences(text)
print(f"Number of sentences: {len(sentences):,}")
print(f"Number of words: {sum(len(s) for s in sentences):,}")
wordfreq = word_frequencies(sentences)
print(f"Vocabulary size: {len(wordfreq):,}")
print(wordfreq['cthulhu'])
print("50 most common words:", wordfreq.most_common(50))
print("50 pretty common words:", wordfreq.most_common(1000)[-50:])  # least common of the top 1000
model = train_w2v(sentences)

neighbors(model, ['cthulhu', 'nyarlathotep', "language", "dread", "tortured"]) #He didn't actually use "Euclidean" as often as I thought.