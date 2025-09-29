# Lyrics Clustering: Bag of Words vs. Embeddings

This project explores how different text representation methods affect the clustering of song lyrics.  
We compare three approaches:

1. **Bag of Words (TF-IDF)**
2. **Skip-Gram Word2Vec**
3. **Pretrained Sentence Embeddings (MiniLM)**

---

## Project Objective
The goal is to understand how lyrics can be grouped into themes (e.g., love songs, rap/explicit, holiday songs) using different text encoding methods.  
We measure cluster quality using **Silhouette Score** and compare the interpretability of each method.

---

## Methods

### 1. Bag of Words (TF-IDF)
- Represents lyrics as word counts.  
- Captures frequency but not meaning.  
- Example: `"love, baby, heart, forever"` → `love:1, baby:1, heart:1, forever:1`

### 2. Skip-Gram (Word2Vec)
- Trains word embeddings on the dataset.  
- Learns which words appear together and captures semantic similarity.  
- Example: `"love"` is close to `"heart"`, `"money"` is close to `"cash"`.

### 3. Pretrained Sentence Embeddings (MiniLM)
- Uses a pretrained transformer model.  
- Converts each lyric into a vector that captures its deeper meaning.  
- Example: `"sad"` and `"cry"` are related even if not present together in the dataset.

---

## Results

### Silhouette Scores
| Method                 | Silhouette Score |
|-------------------------|------------------|
| Bag of Words (TF-IDF)   | 0.016            |
| Skip-Gram (Word2Vec)    | 0.120            |
| Pretrained (MiniLM)     | 0.114            |

### Interpreted Clusters
- **Love/Relationships**: “heart, love, baby, forever”  
- **Rap/Explicit**: “money, cash, bitch, flex”  
- **K-Pop Themes**: “jungkook, jimin, rm, suga”  
- **Holiday/Christmas Songs**: “christmas, merry, snow, mistletoe”  
- **Artist-Specific**: Charlie Puth, Justin Bieber, Cardi B, etc.

---

##  How to Run
1. Open the notebook in Google Colab or Jupyter.
2. Place your dataset (CSV files with a `Lyric` column) in the correct folder.
3. Run all cells:
   - Preprocessing  
   - BoW (TF-IDF)  
   - Skip-Gram (Word2Vec)  
   - Pretrained Embeddings (MiniLM)  
   - Comparison & Visualization  
4. Check the output clusters and silhouette scores.

---

## Key Takeaways
- **BoW** is simple but weak, as it only counts words.  
- **Skip-Gram** learns useful themes directly from the dataset.  
- **Pretrained embeddings** perform strongly by leveraging external knowledge.  

Best performance: **Skip-Gram Word2Vec (0.120 Silhouette)**.

---

## Example Cluster Interpretations
**Skip-Gram Clusters**:  
- Cluster 0 - Love/Emotional Songs  
- Cluster 1 - K-Pop (BTS/Jungkook)  
- Cluster 2 - Relationship/Struggles  
- Cluster 3 - Miscellaneous/Noise  
- Cluster 4 - Rap/Explicit (Cardi B / Nicki Minaj)

---

## Future Work
- Trying larger datasets for better Word2Vec training.  
- Experiment with other pretrained models (BERT, RoBERTa).  
- Use topic modeling (LDA) for comparison.  
- Visualize clusters with t-SNE or UMAP for better interpretability.  

---
