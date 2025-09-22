# Do Meanings Survive Tokenization?
Compare semantic consistency across tokenization methods by training embeddings over the **same corpus** and measuring whether word neighborhoods and similarity structure persist.

## Quick Start
```bash
# 1) Clone / unzip this folder
# 2) Create and activate a venv (recommended)
python -m venv .venv && . .venv/bin/activate  # (Windows: .\.venv\Scripts\activate)
pip install -r requirements.txt

# 3) Add a corpus
# Put a UTF-8 text file at: data/corpus.txt  (e.g., Project Gutenberg #100)

# 4) Train embeddings for all tokenizations (word, char, subword)
python -m src.train_embeddings --vector-size 200 --window 5 --epochs 5 --min-count 5

# 5) Compare spaces (neighbors, rank correlation, Jaccard overlap)
python -m src.compare_spaces --anchors bank apple virus port model crash key stream root cloud kernel attack --topk 10

# 6) Visualize (UMAP/TSNE + similarity heatmaps)
python -m src.visualize --anchors bank apple virus port
```

## What this project does
- Tokenizes the same corpus **three ways**: word-level, character-level, and BPE subword.
- Trains embeddings per method (Word2Vec / FastText for words; Word2Vec for char/subword tokens).
- Compares the resulting spaces using:
  - **Nearest neighbor overlap** (Jaccard@k)
  - **Rank correlations** (Spearman ρ) of similarity profiles
  - Optional **Orthogonal Procrustes** alignment residuals
- Visualizes 2-D geometry with UMAP/TSNE and anchor similarity heatmaps.

## Files
- `src/prepare_data.py` – loads and normalizes raw text.
- `src/tokenize.py` – word, char, and BPE tokenizers (trains BPE if missing).
- `src/train_embeddings.py` – trains embeddings for each tokenization.
- `src/compare_spaces.py` – computes metrics across spaces.
- `src/visualize.py` – plots UMAP/TSNE and heatmaps.
- `results/` – saved models, metrics, and figures.

## Notes
- **Corpus**: Provide a sizable corpus (≥10–20MB) for decent neighborhoods.
- **Anchors**: Choose domain-relevant terms for your class (security, malware, etc.).
- **Reproducibility**: We set seeds where practical, but stochastic algorithms vary slightly.
