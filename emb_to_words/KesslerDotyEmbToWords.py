import numpy as np

def load_word_embeddings(file_path):
    word_vectors = {}
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(list(map(float, parts[1:])))
            word_vectors[word] = vector
    return word_vectors

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def find_similar_words(target_word, word_vectors, top_n=5):
    if target_word not in word_vectors:
        return []
    target_vec = word_vectors[target_word]
    similarities = []
    for word, vec in word_vectors.items():
        if word == target_word:
            continue
        sim = cosine_similarity(target_vec, vec)
        similarities.append((word, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# === Main Execution ===
file_path = 'sample_vectors.txt'
word_vectors = load_word_embeddings(file_path)


target_words = ['economy', 'baseball', 'music', 'computer', 'is' ]

for word in target_words:
    print(f"\n Similar words to: **{word}**")
    similar = find_similar_words(word, word_vectors, top_n=5)
    for neighbor, score in similar:
        print(f"  {neighbor:<15} → similarity: {score:.4f}")



print(f"\n 5 Word Sentence starting with economy")
similar = find_similar_words(word, word_vectors, top_n=5)

seed_word = 'economy'
steps = 5
current_word = seed_word
chain = []

for _ in range(steps):
    results = find_similar_words(current_word, word_vectors, top_n=5)
    if len(results) < 2:
        break
    next_word = results[1][0]
    chain.append(next_word)
    current_word = next_word

print(" ".join(chain))
