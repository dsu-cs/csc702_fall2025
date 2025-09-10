import math
import argparse
import fileinput
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# This function builds a dictionary based on the contents of an input file. Each line in the input is expected to be a word followed by a vector of float values.
# The word in the input is used as the key and the vector is used as the value in the dictionary key-value pair.
def buildDictionary(inFile):
    newDict = {}
    with open(inFile, "r") as file:
        for line in file:
            vec = line.split()
            for i in range(1,len(vec)):
                vec[i]=float(vec[i])
            newDict[vec[0]] = vec[1:]
    return newDict

# This function calculates the distance between two words in the dictionary.
# The parameters are the vectors that are associated with each of the words in the dictionary.
def distanceBetween(wordOne, wordTwo):
    dsqrd = 0
    for x1, x2 in zip(wordOne, wordTwo):
        dsqrd += (x1-x2) ** 2
    d = math.sqrt(dsqrd)
    return d

# This function chooses the next word from those available within jumping distance.
# The parameters are the current word in the dictionary, the target word, the words in the dictionary that are within jumping distance, and the full dictionary.
def jump(curWord, lastWord, jumpWords, dict):
    nextWord = curWord
    curDist = distanceBetween(dict[nextWord], dict[lastWord])
    for word in jumpWords:
        if word == lastWord:
            nextWord = lastWord
            return nextWord
        else:
            d = distanceBetween(jumpWords[word], dict[lastWord])
            if d < curDist:
                nextWord = word
                curDist = d
    return nextWord

# This function finds the words in the dictionary that are within jumping distance.
# The parameters are the current word in the dictionary, the maximum distance that can be traversed at once, and the full dictionary.
def findJumpable(curWord, maxJump, dict):
    jumpWords = {}
    for word in dict:
        if dict[curWord] == dict[word]:
            continue
        d = distanceBetween(dict[curWord], dict[word])
        if d <= maxJump:
            jumpWords[word] = dict[word]
    return jumpWords

# This function finds the number of jumps between starting word and the target word. If verbose is true, it will print out the words in the dictionary as it jumps.
# The parameters are the starting word in the dictionary, the target word, the maximum distance that can be traversed at once, the full dictionary, and a boolean that decides if each word that is jumped to should be printed.
def findJumps(firstWord, lastWord, maxJump, dict, verbose):
    jumps = 0
    finalDistance = distanceBetween(dict[firstWord], dict[lastWord])
    curWord = firstWord
    if firstWord == lastWord:
        print("No jumps are needed.")
        return 0
    d = distanceBetween(dict[firstWord], dict[lastWord])
    if d <= maxJump:
        print("There is only 1 jump from {} to {} with a maximum jump distance of {}.".format(firstWord, lastWord, maxJump))
        return 1
    else:
        while curWord != lastWord:
            if verbose:
                print(curWord)
            jumpWords = findJumpable(curWord, maxJump, dict)
            curWord = jump(curWord, lastWord, jumpWords, dict)
            jumps += 1
        print("The distance between {} and {} is {}. There are {} jumps between {} and {} with a maximum jump distance of {}.".format(firstWord, lastWord, finalDistance, jumps, firstWord, lastWord, maxJump))
        return jumps

# This function finds the consine similarity of two vectors.
# The parameters are the two vectors used for the calculations.    
def CalcCosineSim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# This function findest the closest word to the target in our dictionary based on the cosine similarity.
# The parameters are the target word itself and the full dictionary.
def FindClosestWord(targetWord, vectorDict):
    maxSim = -1
    vector = vectorDict[targetWord]
    closestWord = None
    for word, vec in vectorDict.items():
        sim = CalcCosineSim(vector, vec)
        if sim > maxSim and vector is not vec:
            maxSim = sim
            closestWord = word
    print("The closest word to {} is {}.".format(targetWord,closestWord))
    return closestWord

def parse_args():
    p = argparse.ArgumentParser(description = "This program determines how many jumps are needed to traverse between arbitrary words in a dictionary.")
    p.add_argument("--firstWord", type=str, default="the")
    p.add_argument("--lastWord", type=str, default="aer")
    p.add_argument("--file", type=str, default=".\sample_vectors.txt")
    p.add_argument("--maxJump", type=float, default=1.0)
    p.add_argument("--verbose", type=bool, default=False)
    return p.parse_args()

def main():
    args = parse_args()
    words = buildDictionary(args.file)
    try:
        value = words[args.firstWord]
    except KeyError:
        print("Your starting word does not exist within the dictionary!")
    try:
        value = words[args.lastWord]
    except KeyError:
        print("Your target word does not exist within the dictionary!")
    findJumps(args.firstWord, args.lastWord, args.maxJump, words, args.verbose)
    FindClosestWord(args.lastWord, words)

if __name__ == "__main__":
    main()
