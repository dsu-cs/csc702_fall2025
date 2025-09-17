import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm
from scipy.linalg import orthogonal_procrustes
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

# Get the directory where this script is located
base_dir = Path(__file__).parent

# Files are in the same folder as the script
romeo = base_dir / "juliet.txt"
moby = base_dir / "moby.txt"
modern = base_dir / "modern.txt"

# Ensure NLTK punkt is available
nltk.download("punkt")

# --- Step 1: Preprocess into sentences ---
def preprocess_sentences(text):
    sentences = sent_tokenize(text)
    tokenized = []
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        tokens = [t for t in tokens if t.isalpha()]
        if tokens:
            tokenized.append(tokens)
    return tokenized

# Load texts
with open(romeo, "r", encoding="utf-8") as f:
    romeo_text = f.read()
with open(moby, "r", encoding="utf-8") as f:
    moby_text = f.read()
with open(modern, "r", encoding="utf-8") as f:
    modern_text = f.read()

romeo_sentences = preprocess_sentences(romeo_text)
moby_sentences = preprocess_sentences(moby_text)
modern_sentences = preprocess_sentences(modern_text)

# --- Step 2: Train Word2Vec models ---
def train_model(sentences):
    return Word2Vec(
        sentences=sentences, vector_size=100, window=10,
        min_count=2, workers=4, epochs=100
    ).wv

romeo_kv = train_model(romeo_sentences)
moby_kv = train_model(moby_sentences)
modern_kv = train_model(modern_sentences)

# --- Step 3: Align embeddings ---
def align_to_reference(ref_kv, other_kv):
    common_vocab = list(set(ref_kv.key_to_index).intersection(set(other_kv.key_to_index)))
    ref_matrix = np.array([ref_kv[w] for w in common_vocab])
    other_matrix = np.array([other_kv[w] for w in common_vocab])
    R, _ = orthogonal_procrustes(other_matrix, ref_matrix)
    return {w: other_kv[w] @ R for w in other_kv.key_to_index}

moby_aligned = align_to_reference(romeo_kv, moby_kv)
modern_aligned = align_to_reference(romeo_kv, modern_kv)

# --- Step 4: Similarity helpers ---
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def similarity_matrix(words, kv):
    data = []
    for w1 in words:
        row = []
        for w2 in words:
            if w1 in kv and w2 in kv:
                sim = cosine_similarity(kv[w1], kv[w2])
            else:
                sim = None
            row.append(sim)
        data.append(row)
    return pd.DataFrame(data, index=words, columns=words)

# --- Step 5: Build matrices ---
words_to_check = ["love", "death", "fate", "sea", "whale",
                  "heaven", "lord", "soul", "romance", "cunning"]

romeo_df = similarity_matrix(words_to_check, romeo_kv)
moby_df = similarity_matrix(words_to_check, moby_aligned)
modern_df = similarity_matrix(words_to_check, modern_aligned)

print("\n🔹 Romeo & Juliet similarity matrix:")
print(romeo_df.round(3))
print("\n🔹 Moby-Dick similarity matrix:")
print(moby_df.round(3))
print("\n🔹 A Modern Instance similarity matrix:")
print(modern_df.round(3))
# --- Step 7: Compare shifts ---
def top_shifts(df1, df2, label1, label2, topn=10):
    delta_df = df1 - df2
    shifts = []
    for i in range(len(delta_df.index)):
        for j in range(i+1, len(delta_df.columns)):
            w1, w2 = delta_df.index[i], delta_df.columns[j]
            if pd.notna(delta_df.loc[w1, w2]):
                diff = delta_df.loc[w1, w2]
                shifts.append((w1, w2, diff, abs(diff)))
    shifts_sorted = sorted(shifts, key=lambda x: -x[3])
    print(f"\n🔹 Top {topn} semantic shifts ({label1} vs {label2}):")
    for w1, w2, diff, _ in shifts_sorted[:topn]:
        direction = f"{label1} stronger" if diff > 0 else f"{label2} stronger"
        print(f"{w1:>8} – {w2:<8} | Δ = {diff:.3f} ({direction})")
    return shifts_sorted[:topn]

# Pairwise comparisons
top_shifts(romeo_df, moby_df, "Romeo", "Moby", topn=10)
top_shifts(romeo_df, modern_df, "Romeo", "Modern", topn=10)
top_shifts(moby_df, modern_df, "Moby", "Modern", topn=10)

from sklearn.decomposition import PCA

def plot_pca(words, ref_kv, aligned_kvs, labels):
    # Collect embeddings for all corpora
    vectors, corpus_labels, word_labels = [], [], []
    for word in words:
        for kv, label in zip(aligned_kvs, labels):
            if word in kv:
                vectors.append(kv[word])
                corpus_labels.append(label)
                word_labels.append(word)
    
    # Run PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(10, 8))
    colors = {"Romeo": "red", "Moby": "blue", "Modern": "green"}

    for i, (x, y) in enumerate(reduced):
        plt.scatter(x, y, color=colors[corpus_labels[i]], label=corpus_labels[i] if i < len(words) else "")
        plt.text(x+0.01, y+0.01, f"{word_labels[i]} ({corpus_labels[i]})", fontsize=9)

    # Legend
    handles = [plt.Line2D([0],[0], marker="o", color="w", label=lab,
                          markerfacecolor=col, markersize=10)
               for lab, col in colors.items()]
    plt.legend(handles=handles)
    plt.title("PCA Projection of Word Embeddings Across Corpora")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

# --- Run PCA plot ---
plot_pca(words_to_check, romeo_kv,
         [romeo_kv, moby_aligned, modern_aligned],
         ["Romeo", "Moby", "Modern"])
