# The idea behind what we are doing:

# Queen - Woman = property_x
# Apples - Apple = property_y
# Apple + property_x = word_X ? -- What word is most similar to this resulting vector?

import numpy as np
from numpy.linalg import norm 

# Cosine similarity function
# Thank you "https://www.geeksforgeeks.org/python/how-to-calculate-cosine-similarity-in-python/"

def cosine_similarity(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (norm(a) * norm(b))

# Read in database as numpy vectors

# Each array place is a different word's vector
# with its corresponding values for each dimension.

vectors =  np.loadtxt('sample_vectors.txt', dtype='str', comments=None)

word_labels = vectors[:, 0] 
numeric_vectors = vectors[:, 1:].astype(float)

# print(f'Raw Data: {vectors}\n\n')            # All data
# print(f'Words: {word_labels}\n\n')        # Labels (words)
# print(f'Vectors: {numeric_vectors}\n\n')    # The good stuff :D
# These are very long ^^^


# measure property_x (differences between words)

# x = 8050    # Queen
# y = 1113   # Woman
# z = 898  # Man

x = 8050   # Queen
y = 1113    # Woman
z = 1246    # death

print(f'\n\nMeasuring the difference between Word {x} and Word {y}. These are "{word_labels[x]}" and "{word_labels[y]}".')

# Measure difference between numeric vectors of queen and woman.
property_x = numeric_vectors[x] - numeric_vectors[y]    # Queen - Woman

# Cosine similarity equation:
property_x_cossim = cosine_similarity(numeric_vectors[x], numeric_vectors[y])   

# Now find similarity between the new word with cosine similarity (this is property_X)

# print(f"\n\nThe cos_sim between '{word_labels[x]}' and '{word_labels[y]}' is {property_x_cossim}.\n\n") 

# Take the property that is the difference from these words and add it to a new word.

print(f"\n\nNow adding this property to a new word, {word_labels[z]}, for a new target vector.")

target_vector = numeric_vectors[z] + property_x# print(f'\n\nAdding this property to the word "{word_labels[z]}" results in a new vector: {target_vector}\n\n')  # Vector is very long

# Now find the word that is most similar to this resulting vector.

best_word = -1.0 # closest word vector to target_vector
closest_index = None  # index for best word

for i in range(len(numeric_vectors)):
    if(i == x or i == y or i == z):
        continue  # skip
    else:
        score = cosine_similarity(target_vector, numeric_vectors[i])
        if score > best_word:
            best_word = score
            closest_index = i

# print(f'Best Word Index: {closest_index}')
# print(f'Closest Word Vector: {closest_word_sim}') # This is very long

print(f'\n\nThe word most similar to this new vector is "{word_labels[closest_index]}".')
print(f'''\n\nAnother way of looking at this, the cosine similarity between the new target vector (e.g. (queen - woman) + new word), and the \nrecently found most similar word's ("{word_labels[closest_index]}") vector is {best_word}.''')

print(f"\nSo, {word_labels[x]} - {word_labels[y]} + {word_labels[z]} (according to these embeddings) ≈ {word_labels[closest_index]}.\n\n")