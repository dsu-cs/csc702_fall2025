import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]                 # first column is the word
            vector = np.array(parts[1:], dtype=float)  # rest are floats
            embeddings[word] = vector
    return embeddings

def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

#Example usage
embeddings = load_embeddings("sample_vectors.txt")

#Check size
print("Loaded embeddings:", len(embeddings))

#Look at one word
print("Vector for 'chairman':\n", embeddings["chairman"])
print("Vector dimension:", embeddings["chairman"].shape)

print(calculate_cosine_similarity(embeddings["chairman"], embeddings["american"]))

