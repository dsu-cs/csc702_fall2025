# Shakespeare Word Embeddings Demo

# Shakespeare Context-Poet

This project creates a fun **AI-flavored poetry generator** using Shakespeare’s complete works from [Project Gutenberg (ebook #100)](https://www.gutenberg.org/ebooks/100).

The program:

1. Downloads and cleans Shakespeare’s corpus.
2. Finds **10 random words** that appear in **different contexts**.
3. Shows those words with short context snippets in a text box.
4. Generates a **poem** that uses each word, reflecting its contextual flavors.

---

## Requirements

* **Python 3.9+** (works with 3.8+, but tested with 3.9/3.10)
* **tkinter** (for GUI)
* **requests** (for downloading corpus)

### Installing tkinter

* **Ubuntu/Debian**:

  ```bash
  sudo apt-get install python3-tk
  ```
* **Fedora/RHEL**:

  ```bash
  sudo dnf install python3-tkinter
  ```
* **macOS**: Included with the official python.org installer. If missing:

  ```bash
  brew install python-tk
  ```
* **Windows**: Included by default with python.org installer. To test:

  ```bash
  python -m tkinter
  ```

### Install other dependencies

```bash
pip install requests
```

---

## Usage

Run the script:

```bash
python context_poet.py
```

Steps:

1. Click **“Pick 10 Words”** to extract random words and their contexts.
2. Read their context snippets in the top box.
3. Click **“Write Poem”** to generate a poem using those words.

---

## Example Output

**Context Box (snippet):**

```
— LOVE —
  · Love looks not with the eyes, but with the mind…
  · The course of true love never did run smooth…

— NIGHT —
  · Good night, good night! Parting is such sweet sorrow…
```

**Generated Poem:**

```
A Poem of Many Meanings
by A Little Automaton


01. O love, where smooth leans to mind, thou changest hue.
02. In parting's hush and sorrow's thunder, night walks anew.
...
— fin —
```

<img width="1153" height="882" alt="image" src="https://github.com/user-attachments/assets/261735b3-a6b4-4af8-bbee-107937da0f53" />

---------------------------------------------------------------------------------------------------------------------------------------------
# Second DEMO
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




