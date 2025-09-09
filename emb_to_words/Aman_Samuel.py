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

#Find word that is the furthest away(least cosine similarity) from given word
def find_furthest_word(word):
    furthest_similarity = 1.0
    furthest_word = ''
    for embed in embeddings:
        sim = calculate_cosine_similarity(embeddings[word], embeddings[embed])
        if sim < furthest_similarity:
            furthest_similarity = sim 
            furthest_word = embed
    return furthest_word, furthest_similarity

#Find word that has closest cosine similarity to the similarity parameter value
def find_closest_similarity(word, similarity):
    closest_diff = 1.0
    closest_word = ''
    closest_similarity = 0.0
    for embed in embeddings:
        sim = calculate_cosine_similarity(embeddings[word], embeddings[embed])
        diff = abs(similarity - sim)
        if diff < closest_diff:
            closest_diff = diff
            closest_word = embed
            closest_similarity = sim
    return closest_word, closest_similarity

#Load embeddings
embeddings = load_embeddings("sample_vectors.txt")

#Ask user for word input, ask again if word is not in embeddings
while True:
    word = input("Enter word: ")
    if word in embeddings:
        break
    else:
        print("Word not found in embeddings.")

#Find furthest word and print out cosine similarity results
furthest_word, furthest_similarity = find_furthest_word(word)
print("*" * 60)
print(f"Furthest word from '{word}': {furthest_word} (similarity = {furthest_similarity:.4f})")

#Calculate difference and find 3rd, 2nd, and 1st quartile 
furthest_diff = 1.0 - furthest_similarity
third_qt_similarity  = 1.0 - furthest_diff * 0.75
second_qt_similarity = 1.0 - furthest_diff * 0.50
first_qt_similarity  = 1.0 - furthest_diff * 0.25

#Find word closest to 3rd quartile
third_qt_word, third_qt_sim = find_closest_similarity(word, third_qt_similarity)
print(f"3rd quartile: {third_qt_word} (similarity = {third_qt_sim:.4f})")

#Find word closest to 2nd quartile
second_qt_word, second_qt_sim = find_closest_similarity(word, second_qt_similarity)
print(f"2nd quartile: {second_qt_word} (similarity = {second_qt_sim:.4f})")

#Find word closest to 1st quartile
first_qt_word, first_qt_sim = find_closest_similarity(word, first_qt_similarity)
print(f"1st quartile: {first_qt_word} (similarity = {first_qt_sim:.4f})")

#Print word chain showing all words in order
print(f"Word Chain: {furthest_word} -> {third_qt_word} -> {second_qt_word} -> {first_qt_word} -> {word}")