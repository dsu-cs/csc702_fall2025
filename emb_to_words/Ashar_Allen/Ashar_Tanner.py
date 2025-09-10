import numpy as np
from numpy import dot
from numpy.linalg import norm

# read file for word and stores word's vector
def read(word):
    file = open("sample_vectors.txt", "r")
    for line in file:
        if word in line:    
            try:
                numbers = line.split()[1:]
                number_line = " ".join(numbers)
                wordVector = np.fromstring(number_line, dtype=float, sep=' ')

            except:
                print("Not a number")
            break
    return wordVector

#Vector math
def sub(VectorA, VectorB):
    difference = VectorA - VectorB
    return difference

def add(VectorA, VectorB):
    add = VectorA + VectorB
    return add

def cosSim(VectorA, VectorB):
    cos_sim = dot(VectorA, VectorB)/(norm(VectorA)*norm(VectorB))
    return cos_sim

#main
def main():
    print("Enter words for formula \n")
    print("A - B + C \n")
    a = input("Word for A: ")
    vectora = read(a)


    print("\n")
    b = input("Word for B: ")
    vectorb = read(b)

    subVector = sub(vectora, vectorb)

    print("\n")
    c = input("word for C: ")
    vectorc = read(c)

    addVector = subVector + vectorc
    print(addVector)

    print("\n")
    
main()