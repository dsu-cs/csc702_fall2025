import sentencepiece as spm
import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import argparse
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# This function prepares data from a user-supplied input file. A new file is produced that is properly formatted for SentencePiece.
# The parameter is the string path to the desired input file.
# The output is the string path to the new output file.
'''
def prepareData(inFile):
    internalList = []
    outFile = inFile[:-4] + "TrainingData.txt"
    with open(inFile, "r") as file:
        rawInput = file.read()
        modInput = rawInput.replace("\n", " ")
        for i in sent_tokenize(modInput):
            temp = i + "\n"
            internalList.append(temp)
    with open(outFile, "w") as out:
        out.writelines(internalList)
    return outFile
'''

#This function prepares data from a user-supplied input file.
# The function uses Lemmatization to create an output file that is formatted for SentencePiece
# The parameter is the string path to the desired input file.
# The output is the string path to the new output file
def prepare_lemmatized_data(inFile):
    internalList = []
    outFile = inFile[:-4] + "_lemmatized.txt"
    lemmatizer = WordNetLemmatizer()

    with open(inFile, "r", encoding="utf-8") as file:
        raw = file.read().replace("\n", " ")
        for sentence in sent_tokenize(raw):
            tokens = word_tokenize(sentence)
            lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
            internalList.append(" ".join(lemmatized) + "\n")
    
    with open(outFile, "w", encoding="utf-8") as out:
        out.writelines(internalList)
    return outFile


#This function prepares data from a user-supplied input file.
# The function uses Stemming to create an output file that is formatted for SentencePiece
# The parameter is the string path to the desired input file.
# The output is the string path to the new output file
def prepare_stemmed_data(inFile):
    internalList = []
    outFile = inFile[:-4] + "_stemmed.txt"
    stemmer = PorterStemmer()

    with open(inFile, "r", encoding="utf-8") as file:
        raw = file.read().replace("\n", " ")
        for sentence in sent_tokenize(raw):
            tokens = word_tokenize(sentence)
            stemmed = [stemmer.stem(w) for w in tokens]
            internalList.append(" ".join(stemmed) + "\n")
    
    with open(outFile, "w", encoding="utf-8") as out:
        out.writelines(internalList)
    return outFile


# This function creates a unigram model using the supplied input file. This creates two files: modelName.model and modelName.vocab.
# The modelName parameter is the name of the new model, and inFile is the string path to the input file.
# The output is the .model file. 
def createModel(modelName, inFile):
    spm.SentencePieceTrainer.Train(
        input = inFile,
        model_prefix = modelName,
        vocab_size = 3000,
        model_type = "unigram"
    )
    output = modelName + ".model"
    return output

# fileOne and Two are string paths to the initial input files.
# For this specific use case we will just use fileOne
# modelOne and Two are the names of the two new models.
# testString is the user input string.
def parse_args():
    p = argparse.ArgumentParser(description = "This program tokenizes a text in a unigram model using SentencePiece.")
    p.add_argument("--fileOne", type = str, default = "./Dracula.txt")
    # p.add_argument("--fileTwo", type = str, default = "./Frankenstein.txt")
    p.add_argument("--modelOne", type = str, default = "modelOne")
    p.add_argument("--modelTwo", type = str, default = "modelTwo")
    p.add_argument("--testString", type = str, default = "This is a test sentence.")
    return p.parse_args()

def main():
    args = parse_args()
    # Loads and prepares data, constructs the models.
    # outFileOne = prepareData(args.fileOne)
    # outFileTwo = prepareData(args.fileTwo)
    # modelOne = createModel(args.modelOne, outFileOne)
    # modelTwo = createModel(args.modelTwo, outFileTwo)

    lemma_file = prepare_lemmatized_data(args.fileOne)
    stem_file = prepare_stemmed_data(args.fileOne)
    model_lemma = createModel(args.modelOne, lemma_file)
    model_stem = createModel(args.modelTwo, stem_file)
    
    # Loads the models.
    m1 = spm.SentencePieceProcessor()
    m2 = spm.SentencePieceProcessor()
    m1.load(model_lemma)
    m2.load(model_stem)

    # Encodes the test string as the toekn ids and as the token pieces.
    ids1 = m1.encode_as_ids(args.testString)
    ids2 = m2.encode_as_ids(args.testString)
    p1 = m1.encode_as_pieces(args.testString)
    p2 = m2.encode_as_pieces(args.testString)

    # Prints a comparison of the two models.
    print("Original Text:", args.testString)
    print("Lemmatization Model IDs:", ids1, "Lemmatization Model Pieces:", p1)
    print("Stem Model IDs:", ids2, "Stem Model Pieces:", p2)

if __name__ == "__main__":
    main()
