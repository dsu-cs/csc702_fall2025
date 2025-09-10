import math
import random
import matplotlib.pyplot as plt

def load_word_vectors(filepath, max_words=None):
    word_vectors = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_words is not None and i >= max_words:
                break
            parts = line.strip().split()
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            word_vectors[word] = vector
    print(f"Loaded {len(word_vectors)} word vectors.")
    return word_vectors

def cosine_similarity(vec1, vec2):
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

def blob_game(word_vectors, seed=42, visualize=False):
    words = list(word_vectors.keys())
    if len(words) < 2:
        print("Need at least 2 words to start the game.")
        return []
    random.seed(seed)
    start_words = random.sample(words, 2)
    blobs = [[start_words[0]], [start_words[1]]]
    remaining_words = [w for w in words if w not in start_words]
    print(f"Starting blob game with 2 random blobs: '{start_words[0]}' and '{start_words[1]}', seed={seed}.")

    vectors = word_vectors.copy()
    merge_count = 0
    sizes_history = [[1], [1]]  # Track sizes for each blob

    while remaining_words:
        next_word = remaining_words.pop(0)
        best_blob = None
        best_sim = -1
        for i, blob in enumerate(blobs):
            w = blob[-1]
            sim = cosine_similarity(vectors[w], vectors[next_word])
            if sim > best_sim:
                best_sim = sim
                best_blob = i
        blobs[best_blob].append(next_word)
        merge_count += 1
        # Track sizes
        for i in range(len(blobs)):
            if len(sizes_history) <= i:
                sizes_history.append([len(blobs[i])])
            else:
                sizes_history[i].append(len(blobs[i]))

    print("Blob game complete.")
    blob_sizes = [len(blob) for blob in blobs]
    winner_index = blob_sizes.index(max(blob_sizes))
    print(f"Blob {winner_index} won with {blob_sizes[winner_index]} words.")

    # Visualization
    if visualize:
        for i, history in enumerate(sizes_history):
            plt.plot(history, label=f'Blob {i}')
        plt.xlabel('Steps')
        plt.ylabel('Blob Size')
        plt.title('Blob Growth Over Time')
        plt.legend()
        plt.show()

    return blobs

if __name__ == "__main__":
    vectors = load_word_vectors(
        r"emb_to_words\Lucas-Piyush\sample_vectors.txt",
        max_words=10000
    )
    result_blobs = blob_game(vectors, seed=10, visualize=True)

