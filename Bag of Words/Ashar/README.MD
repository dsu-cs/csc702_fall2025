# Bag of Words

This project demonstrates how to transform sparse ** Bag of Words (BoW) ** into compact **document embeddings** using different techniques. It also benchmarks these embeddings with a simple classifier and
visualizes them.

BowEmbeddings.Py # Main script
README
data/ # Dataset folder
sports/ 
s1.txt
s2.txt
s3.txt
politics/
p1.txt
p2.txt
p3.txt
tech/
t1.txt
t2.txt
t3.txt
'''

Each ".txt" file is treated as a document. The subfolder name is used as the **class label**.


to run the script, python BowEmbeddings.py --data-dir ./data --embed-dims 100 --plot --save

# Arguments

--data-dir : Path to dataset folder (default: ./data).

--embed-dims : Embedding dimensionality (auto-adjusts if dataset is too small).

--plot : Show t-SNE visualization of embeddings.

--save : Save embeddings to CSV (lsa100_embeddings.csv, nmf100_embeddings.csv, etc.).

--no-autoencoder : Skip autoencoder step if PyTorch is not available.

# Output

## Console Results

Accuracy from Logistic Regression (cross-validation) for:

TF-IDF (baseline)

LSA embeddings

NMF embeddings

## Autoencoder embeddings 

Plots (--plot)

2D scatterplots of embeddings via t-SNE, colored by class.

## CSV Files (--save)

Embedding vectors saved in files like:

lsa50_embeddings.csv

nmf50_embeddings.csv

ae50_embeddings.csv



