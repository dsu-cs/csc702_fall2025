from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------
# two FUNctions that mess with the vectors to better grasp the concealed meaning
#
# DimensionUnclimb() reduces our vectors to a single dimension
# inspired by the alphabetical example where cat is more similar to car
# perhaps we would get a better ordered list of words if we chopped down from our higher dimensions
#
# DimesnionMeaning() find the strongest word for each dimension in our vectors
# perhaps we'll find that animalistic dimension I mused about in my discussion post
#---------------------------------------------------------------------------------


embeddings = KeyedVectors.load_word2vec_format(file, binary=False, no_header=True)

#print(embeddings["king"])
#print(embeddings.most_similar("king"))
#print(embeddings.similarity("king", "queen"))


def DimensionUnclimb():
    #what if we shrunk all our word vectors to one dimension?????
    #could we order the words better than alphabetical?
    words = embeddings.index_to_key
    vectors = np.array([embeddings[w] for w in words])

    #dimension reduction thing
    pca = PCA(n_components=1)
    reduced = pca.fit_transform(vectors)
    wordPositions = {w: reduced[i,0] for i,w in enumerate(words)}

    #seems the words go on a political to commerical spectrum
    #democratic, republican, murder, protest on one end
    #coca cola, inc, earnings, quarterly on the other end
    #maybe not the best for comparing immediate adjacency, but things like company names seem to be fairly close. 
    sortedWords = sorted(wordPositions.items(), key=lambda x: x[1])
    print(sortedWords)


def DimensionMeaning():
    #which word best describes each dimension
    #is there a dog dimension?????
    words = embeddings.index_to_key
    vectors = np.array([embeddings[w] for w in words])

    #find the word with the max value for each dim
    maxIndex = np.argmax(vectors, axis=0)
    maxWords = [words[i] for i in maxIndex]

    #max value
    maxValues = vectors[maxIndex, range(vectors.shape[1])]

    #mostly boring dimesions
    #we have a "the" and "$" dimension...
    #notable dimensions include the "father" and "whiskey" dimensions!
    #"king" has 0.6 on the "the" axis, 0.2 on the "father" axis, and 0.2 on the "gte"? axis
    #"the" is probably just polluting all the vectors as a filler word or something
    #perhaps something to do with it having 0.94 as its max value compared to most other dimensions only going up to 0.6
    #0.2 for "father" is fairly high since most dimesnions are close to 0 or negative
    #makes sense since both are masculine words
    for d in range(vectors.shape[1]):
        print("Dimension ", d, ": ", maxWords[d], "(value = ", maxValues[d], ")")
    
#DimensionUnclimb()
DimensionMeaning()
print(embeddings["king"]) 

#print(embeddings.most_similar("dog"))
# returns: [('brilliant', 0.8089377880096436), ('wife', 0.782331109046936), ('dream', 0.7809006571769714), ('wears', 0.7463341355323792)...

#---------------------------------------------------------------------------------
# NOTES
# I do not like the data since there seemed to be a massive lack of dogs
# my toying with the similarity function left me disapointed
# "brilliant"? "wife"? "dream"? does that even make sense for "dog"????
# perhaps we would get cooler results with better vectors
#---------------------------------------------------------------------------------