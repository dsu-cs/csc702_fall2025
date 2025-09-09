# Queen - Woman = property_1

# Apples - Apple = property_2

# Apple + property1 = X ? -- What word is most similar?



# Import database and libraries
import numpy as np
from numpy.linalg import norm   # cosine similarity

# Cosine similarity function
# Thank you "https://www.geeksforgeeks.org/python/how-to-calculate-cosine-similarity-in-python/"

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Read in database as numpy vectors

# Each array place is a different word's vector
# with its corresponding values for each dimension.

vectors =  np.loadtxt('sample_vectors.txt', dtype='str', comments=None)

word_labels = vectors[:, 0]  # seperate word labels from remainder of vectors
numeric_vectors = vectors[:, 1:].astype(float)

# Troubleshooting :/

# print(vectors)            # All data
# print(word_labels)        # Labels (words)
# print(numeric_vectors)    # The good stuff :D

# measure properties (differences between words)

            # # Example

            # x = 0
            # y = 3

            # print(f'\n\nMeasuring the difference between Word {x} and Word {y}. These are "{word_labels[x]}" and "{word_labels[y]}".\n\n')

            # # Measure cosine similarity between numeric vectors of 1st and 4th words.
            # temp_sim = cosine_similarity(numeric_vectors[x], numeric_vectors[y])   

            # print(f'\nDifference between the two words is approximately... {temp_sim}.')
            # print('\nUpon manual analysis, this is a good representation as the vectors are truly similar.')
            # print('\nNow, lets find two dissimilar words.\n\n')

            # x = 0
            # y = 6833

            # print(f'\n\nMeasuring the difference between Word {x} and Word {y}. These are "{word_labels[x]}" and "{word_labels[y]}".\n\n')

            # # Measure cosine similarity between numeric vectors of 1st and 6833rd words.
            # temp_sim = cosine_similarity(numeric_vectors[x], numeric_vectors[y])   

            # print(f'\nDifference between the two words is approximately... {temp_sim}.')
            # print('\nUpon manual analysis, this is a good representation as the vectors are much less similar.')


# Find differences between words with cosine similarity (this is property_X)

x = 933
y = 932
cosine_similarity(numeric_vectors[x], numeric_vectors[y])  # shows difference between words x and y.
print(f"\n\nThe difference between '{word_labels[x]}' and '{word_labels[y]}' is approximately {cosine_similarity(numeric_vectors[x], numeric_vectors[y])}.\n\n")

# Take the property that is the difference from this similarity search and apply it to other words.

x = 933
y = 932

temp_property = cosine_similarity(numeric_vectors[x], numeric_vectors[y])  # shows difference between words x and y.