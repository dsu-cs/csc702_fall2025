import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm
from scipy.linalg import orthogonal_procrustes


romeo = "csc702_fall2025\words_to_emb\Lucas\juliet.txt"
moby = "csc702_fall2025\words_to_emb\Lucas\moby.txt"
# Make sure tokenizer is available
nltk.download("punkt")

# --- Step 1: Load local files ---
with open(romeo, "r", encoding="utf-8") as f:
    romeo_text = f.read()

with open(moby, "r", encoding="utf-8") as f:
    moby_text = f.read()

# --- Step 2: Preprocess into sentences ---
def preprocess_sentences(text):
    sentences = sent_tokenize(text)
    tokenized = []
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        tokens = [t for t in tokens if t.isalpha()]  # keep words only
        if tokens:
            tokenized.append(tokens)
    return tokenized

romeo_sentences = preprocess_sentences(romeo_text)
moby_sentences = preprocess_sentences(moby_text)

# --- Step 3: Train Word2Vec models ---
romeo_model = Word2Vec(
    sentences=romeo_sentences, vector_size=100, window=10,
    min_count=2, workers=4, epochs=100
)
moby_model = Word2Vec(
    sentences=moby_sentences, vector_size=100, window=10,
    min_count=2, workers=4, epochs=100
)

romeo_kv = romeo_model.wv
moby_kv = moby_model.wv

# --- Step 4: Align embeddings with Procrustes ---
common_vocab = list(set(romeo_kv.key_to_index).intersection(set(moby_kv.key_to_index)))
romeo_matrix = np.array([romeo_kv[w] for w in common_vocab])
moby_matrix = np.array([moby_kv[w] for w in common_vocab])

R, _ = orthogonal_procrustes(moby_matrix, romeo_matrix)
moby_aligned = {w: moby_kv[w] @ R for w in moby_kv.key_to_index}

# --- Step 5: Helpers ---
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def most_similar_aligned(target, aligned_dict, topn=5):
    if target not in aligned_dict:
        return []
    target_vec = aligned_dict[target]
    sims = {}
    for w, vec in aligned_dict.items():
        if w != target:
            sims[w] = cosine_similarity(target_vec, vec)
    return sorted(sims.items(), key=lambda x: -x[1])[:topn]

# --- Step 6: Compare key words ---
words_to_check = ["love", "death", "fate", "sea", "whale",
                  "heaven", "lord", "soul", "romance", "cunning"]

for word in words_to_check:
    if word in romeo_kv and word in moby_aligned:
        print(f"\n🔹 Word: {word}")
        print("Romeo & Juliet neighbors:")
        print(romeo_kv.most_similar(word, topn=5))
        
        print("Moby-Dick neighbors:")
        print(most_similar_aligned(word, moby_aligned, topn=5))
    else:
        print(f"\n⚠️ Word '{word}' not in both vocabularies.")
# Compare how a target word relates to the same set of words in both corpora

def compare_word_similarities(target, candidates, kv1, kv2_aligned):
    results = []
    if target not in kv1 or target not in kv2_aligned:
        print(f"⚠️ '{target}' not in both vocabularies.")
        return results
    
    for cand in candidates:
        if cand in kv1 and cand in kv2_aligned:
            sim1 = cosine_similarity(kv1[target], kv1[cand])
            sim2 = cosine_similarity(moby_aligned[target], moby_aligned[cand])
            results.append((cand, sim1, sim2))
    return results

# Example usage
target_word = "love"
candidates = ["death", "fate", "sea", "whale", "heaven", "soul", "lord"]

comparisons = compare_word_similarities(target_word, candidates, romeo_kv, moby_aligned)

print(f"\n🔹 Similarity comparisons for '{target_word}':")
for cand, sim1, sim2 in comparisons:
    print(f"{cand:>10} | Romeo: {sim1:.3f} | Moby: {sim2:.3f}")

import pandas as pd

def similarity_matrix(words, kv1, kv2_aligned):
    data_romeo = []
    data_moby = []
    
    for w1 in words:
        row_r = []
        row_m = []
        for w2 in words:
            if w1 in kv1 and w2 in kv1:
                sim_r = cosine_similarity(kv1[w1], kv1[w2])
            else:
                sim_r = None
            if w1 in kv2_aligned and w2 in kv2_aligned:
                sim_m = cosine_similarity(moby_aligned[w1], moby_aligned[w2])
            else:
                sim_m = None
            row_r.append(sim_r)
            row_m.append(sim_m)
        data_romeo.append(row_r)
        data_moby.append(row_m)
    
    df_romeo = pd.DataFrame(data_romeo, index=words, columns=words)
    df_moby = pd.DataFrame(data_moby, index=words, columns=words)
    return df_romeo, df_moby

# Run the comparison
words_to_check = ["love", "death", "fate", "sea", "whale",
                  "heaven", "lord", "soul", "romance", "cunning"]

romeo_df, moby_df = similarity_matrix(words_to_check, romeo_kv, moby_aligned)

print("\n🔹 Romeo & Juliet similarity matrix:")
print(romeo_df.round(3))

print("\n🔹 Moby-Dick similarity matrix:")
print(moby_df.round(3))


