import numpy as np

# ----------------------------
# 1️ Load embeddings
# ----------------------------
embeddings = {}
with open(r"C:\Users\komal\Documents\projects\Project-2-702\sample_vectors.txt", "r") as f: # Update path as needed
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vector = np.array(list(map(float, parts[1:])))
        embeddings[word] = vector

print(f"Loaded {len(embeddings)} words.")

# ----------------------------
# 2️ Cosine similarity (safe)
# ----------------------------
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return -1
    return np.dot(vec1, vec2) / (norm1 * norm2)

# ----------------------------
# 3️ Find most similar word
# ----------------------------
def most_similar(vector, embeddings, exclude=[]):
    best_word = None
    best_sim = -1
    for word, vec in embeddings.items():
        if word in exclude:
            continue
        sim = cosine_similarity(vector, vec)
        if sim > best_sim:
            best_sim = sim
            best_word = word
    return best_word, best_sim

# ----------------------------
# 4 Property vector arithmetic
# ----------------------------

# Property 1: Queen - Woman
property_1 = embeddings['queen'] - embeddings['woman']
print("Property 1: Queen - Woman")

# Property 2: Apples - Apple
property_2 = embeddings['apples'] - embeddings['apple']
print("Property 2: Apples - Apple")

# Example: Apple + Property 1
vector = embeddings['apple'] + property_1
word, sim = most_similar(vector, embeddings, exclude=['apple', 'queen', 'woman'])
print(f"Apple + property_1 = {word} (similarity: {sim:.4f})")

# Example: Apple + Property 2
vector2 = embeddings['apple'] + property_2
word2, sim2 = most_similar(vector2, embeddings, exclude=['apple', 'apples'])
print(f"Apple + property_2 = {word2} (similarity: {sim2:.4f})")

# ----------------------------
# 5 Optional: Top 5 similar words
# ----------------------------
def top_n_similar(vector, embeddings, n=5, exclude=[]):
    sims = []
    for word, vec in embeddings.items():
        if word in exclude:
            continue
        sims.append((word, cosine_similarity(vector, vec)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:n]

print("\nTop 5 predictions for Apple + property_1:")
for w, s in top_n_similar(embeddings['apple'] + property_1, embeddings, exclude=['apple','queen','woman']):
    print(f"{w} ({s:.4f})")