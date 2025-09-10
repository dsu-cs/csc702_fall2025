# Word Embedding CLI  

This tool provides a simple command-line interface (CLI) to explore **word embeddings** stored in a text file. You can inspect vocabulary size, find similar words, solve analogies, and visualize embeddings in 2D with PCA.  

---
## 🚀 Usage
Run the CLI with:  
```bash
python embeddings.py --file sample_vectors.txt <command> [options]
```
---

## 🛠 Commands

### 1. Check dimensions
```bash
python embeddings.py --file sample_vectors.txt dim
```
Shows the **vocabulary size** and the **vector dimension**.  
Example output:
```
Vocab size: 9998
Vector dim: 40
```

---

### 2. View vocabulary
```bash
python embeddings --file sample_vectors.txt vocab --top 20
```
Lists the first N words in the vocabulary (default = all words).  
Useful for checking what’s inside your embeddings file.  

---

### 3. Find nearest neighbors
```bash
python embeddings.py --file sample_vectors.txt neighbors --word company --top 10
```
Finds the **most similar words** to a given word, based on cosine similarity.  
Example output:
```
Nearest to 'company':
  holding              cos=0.8893
  utility              cos=0.7682
  ...
```

---

### 4. Solve analogies
```bash
python embeddings.py --file sample_vectors.txt analogy --a king --b man --c woman --top 5
```
Solves the classic word analogy **“a is to b as c is to ?”** using vector arithmetic:  
b - a + c ≈ d  
Example: *king – man + woman ≈ queen*  

---

### 5. PCA visualization
```bash
python embeddings.py --file sample_vectors.txt pca --words company market million year this --out pca.png
```
Performs **Principal Component Analysis (PCA)** to reduce vectors to 2D and plots the selected words on a scatterplot. The plot is saved as an image (`pca.png`).  

---


## 📌 Notes
- All similarity is based on **cosine similarity** of normalized vectors.  
- Words not found in the vocabulary will be skipped (with a warning).  
- PCA plots require "matplotlib".  

### Second Approch
