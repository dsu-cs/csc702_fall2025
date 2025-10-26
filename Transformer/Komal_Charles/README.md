---

# mini_transformer (IMDB Sentiment)

This repository contains a compact PyTorch Transformer classifier trained on the IMDB movie-review dataset.
It provides an end-to-end sentiment-classification pipeline for experimentation, rapid iteration, and interpretability through attention visualization.

---

## Contents

* IMDB_Transformer_Sentiment.ipynb – Complete Jupyter notebook containing:

  * Data loading and preprocessing
  * Regex tokenization and vocabulary creation
  * Dataset and DataLoader setup with dynamic padding
  * Transformer model definition (embedding, sinusoidal positional encoding, encoder layers)
  * Training and evaluation loops
  * Learning-rate scheduling, early stopping, and gradient clipping
  * Attention-map extraction and visualization
* outputs/ – Folder containing saved model weights and vocabulary:

  * best_model.pt – Best validation checkpoint
  * vocab.json – Serialized vocabulary dictionary
* data/imdb_reviews.csv – CSV dataset (generated automatically or added manually)

---

## Requirements

Create and activate a Python environment (Python 3.8 or higher is recommended), then install dependencies:

```bash
pip install torch torchvision torchaudio pandas scikit-learn matplotlib seaborn datasets
```

Optional: if you use a requirements.txt file, install everything with:

```bash
pip install -r requirements.txt
```

---

## Dataset

You can either:

1. **Download automatically from HuggingFace:**
2. **Or provide your own CSV** in the same format:


---

## Running the Notebook

1. Open IMDB_Transformer_Sentiment.ipynb in Jupyter or VS Code.
2. Adjust configuration parameters near the top (SUBSET, MAX_LEN, EMBED_DIM, EPOCHS, etc.).
3. Run all cells sequentially.
4. The notebook will:

   * Train a Transformer encoder model on IMDB reviews
   * Print training and validation metrics per epoch
   * Save the best model and vocabulary to ./outputs/

### Subsampling for Quick Tests

For rapid debugging or limited hardware, set:

```python
SUBSET = 0.1    # use 10% of dataset
# or
N_SAMPLES = 5000
```

To train on the full dataset, set both to None.

---

## Model Architecture

* **Embedding Layer** with padding index
* **Sinusoidal Positional Encoding** for sequence order
* **Transformer Encoder** with configurable layers and attention heads
* **Masked Mean Pooling** over non-PAD tokens
* **Dropout + Linear Classifier** for sentiment output

Training uses **AdamW** optimizer, **CrossEntropyLoss**, and gradient clipping.
Optional features include **ReduceLROnPlateau** scheduling and **early stopping** based on validation accuracy.

---

## Visualizing Self-Attention

The notebook captures attention weights from each encoder layer by explicitly calling MultiheadAttention with need_weights=True.

Use the provided function:

```python
visualize_attention_on_batch(
    model,
    val_loader,
    batch_index=0,
    layer_index=0,
    head_index=0,
    max_tokens_to_show=50
)
```

### Interpreting Attention

* Attention tensor shape: (B, H, T, S) = batch, heads, query positions, source positions.
* Each row (query position) sums to 1 across source positions (softmax weights).
* Heatmaps show how tokens attend to one another.

---

## Example Results

| Metric                    | Compact Config (2 Layers × 2 Heads) |
| ------------------------- | ----------------------------------- |
| Train Accuracy            | ~88–90%                             |
| Validation Accuracy       | ~85–88%                             |
| Test Accuracy             | ~84–87%                             |
| Runtime (GPU, 10% subset) | ≈ 3 minutes                         |

Performance may vary depending on dataset size, hardware, and hyperparameters.

---
