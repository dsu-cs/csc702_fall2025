import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# file_path = os.path.join('emb_to_words', 'sample_vectors.txt')

def read_word_vectors(filepath):
    word_vectors = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            try:
                vector = [float(x) for x in parts[1:]]
                word_vectors[word] = vector
            except ValueError:
                continue  # skip lines that don't parse correctly
    return word_vectors

vectors = read_word_vectors('emb_to_words/sample_vectors.txt')

words = list(vectors.keys())
X = np.array(list(vectors.values()))

# Dimensionality reduction
n_components = 10  # Change this to your desired reduced dimension
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X)
# X_reduced = normalize(X_reduced)

# Map words to reduced vectors
reduced_vectors = {word: vec for word, vec in zip(words, X_reduced)}

def find_similar_words(target_word, word_vectors, top_n=5):
    if target_word not in word_vectors:
        return []
    target_vec = np.array(word_vectors[target_word]).reshape(1, -1)
    all_words = list(word_vectors.keys())
    all_vecs = np.array(list(word_vectors.values()))
    similarities = cosine_similarity(target_vec, all_vecs).flatten()
    similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]# Exclude the target word itself
    similar_words = [(all_words[i], similarities[i]) for i in similar_indices]
    return similar_words

similar_words = find_similar_words('korean', vectors, top_n=5)
print(f"Top 5 similar words original vector:")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
similar_words_reduced = find_similar_words('korean', reduced_vectors, top_n=5)
print(f"Top 5 similar words reduced vectors:")
for word, score in similar_words_reduced:
    print(f"{word}: {score:.4f}")





# import numpy as np

# def load_word_embeddings(file_path):
#     word_vectors = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             parts = line.strip().split()
#             word = parts[0]
#             vector = np.array(list(map(float, parts[1:])))
#             word_vectors[word] = vector
#     return word_vectors

# def cosine_similarity(vec1, vec2):
#     norm1 = np.linalg.norm(vec1)
#     norm2 = np.linalg.norm(vec2)
#     if norm1 == 0 or norm2 == 0:
#         return 0.0
#     return np.dot(vec1, vec2) / (norm1 * norm2)

# def find_similar_words(target_word, word_vectors, top_n=20):
#     if target_word not in word_vectors:
#         return []
#     target_vec = word_vectors[target_word]
#     similarities = []
#     for word, vec in word_vectors.items():
#         if word == target_word:
#             continue
#         sim = cosine_similarity(target_vec, vec)
#         similarities.append((word, sim))
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return similarities[:top_n]


# file_path = 'emb_to_words/sample_vectors.txt'
# word_vectors = load_word_embeddings(file_path)

# # Define your 5 target words here
# target_words = ['economy', 'inflation', 'growth', 'market', 'jobs']
# for word in target_words:
#     print(f"\n🔍 Similar words to: **{word}**")
#     similar = find_similar_words(word, word_vectors, top_n=20)
#     for neighbor, score in similar:
#         print(f"  {neighbor:<15} → similarity: {score:.4f}")



#     print(f"\n 5 Word Sentence starting with economy")
#     similar = find_similar_words(word, word_vectors, top_n=5)

#     seed_word = 'economy'
#     steps = 5
#     current_word = seed_word
#     chain = []

#     for _ in range(steps):
#         results = find_similar_words(current_word, word_vectors, top_n=5)
#         if len(results) < 2:
#             break
#         next_word = results[1][0]
#         chain.append(next_word)
#         current_word = next_word

#     print(" ".join(chain))