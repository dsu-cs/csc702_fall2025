# Shakespeare Word Embeddings Demo

This project demonstrates **Word Embeddings** using Shakespeare’s complete works from [Project Gutenberg (ebook #100)](https://www.gutenberg.org/ebooks/100). It trains a **Word2Vec** model on the text and provides interactive examples like nearest neighbors, analogies, and t-SNE visualizations.

## Features
- 📥 **Automatic download** of Shakespeare’s corpus
- 🧹 **Text preprocessing & tokenization**
- 🧠 **Word2Vec training** (skip-gram)
- 🔍 **Word similarity queries** (nearest neighbors)
- 🧩 **Word analogies** (e.g., `king - man + woman ≈ queen`)
- 🚫 **Odd-one-out detection**
- 📊 **t-SNE visualization** of embeddings
- 📂 **TensorBoard Projector TSVs** for exploration

## Requirements
The script auto-installs missing packages. Manually, you can install them with:
```bash
pip install gensim tqdm scikit-learn matplotlib numpy requests
```

## Usage
Run the script directly:
```bash
python shakespeare_embeddings.py
```

By default, it will:
- Train (or load) the model
- Show neighbors for some Shakespearean words
- Solve one analogy
- Find an odd word out
- Create a t-SNE plot of 400 words at `outputs/embedding_tsne.png`

### Examples
- Nearest neighbors:
  ```bash
  python shakespeare_embeddings.py --neighbors king queen love romeo
  ```
- Analogy:
  ```bash
  python shakespeare_embeddings.py --analogy king man woman
  ```
- Odd one out:
  ```bash
  python shakespeare_embeddings.py --oddone romeo hamlet banana mercutio
  ```
- Plot top 500 frequent words:
  ```bash
  python shakespeare_embeddings.py --plot 500
  ```

### Custom Training Parameters
You can override defaults:
```bash
python shakespeare_embeddings.py --vector-size 300 --window 7 --min-count 5 --epochs 15
```

## Output Files
- **data/shakespeare.txt** → raw corpus
- **models/shakespeare.w2v** → trained model
- **outputs/embedding_tsne.png** → 2D visualization
- **outputs/embeddings.tsv / metadata.tsv** → for TensorBoard Projector

## Example Results
- Nearest neighbors of *king* might include *queen*, *duke*, *emperor*, etc.
- Analogy `king - man + woman` ≈ *queen*
- Odd one out of `[romeo, hamlet, banana, mercutio]` → *banana*

## License
- Shakespeare’s works are public domain.
- Code released under MIT License.

