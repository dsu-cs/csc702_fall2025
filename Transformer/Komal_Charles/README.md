# mini_transformer (IMDB Sentiment)

 contains PyTorch Transformer classifier trained on the IMDB movie-review dataset. It is designed for experiments, quick iteration, and attention-visualization.

## Contents

- `Transfomer.ipynb` - Jupyter notebook with the full pipeline: data loading, tokenization, vocabulary building, dataset/dataloader, model definition, training loop, evaluation, and attention capture/visualization cells.
- `data/IMDB Dataset.csv` 

## Requirements

Create and activate a Python environment (recommended Python 3.8+). Install dependencies:

```powershell
py -3.13 -m pip install -r requirements.txt
```

If you don't have `requirements.txt`, install the essentials:

```powershell
py -3.13 -m pip install torch torchvision pandas scikit-learn matplotlib seaborn optuna datasets
```


 Optionally set `SUBSET` to a small number (e.g., 5000) for fast debugging.

## Run training from notebook

- Set `SUBSET = None` to use full dataset.
- Adjust hyperparameters near top (BATCH_SIZE, EPOCHS, LR).


## Visualizing self-attention

The notebook contains cells that capture attention weights by calling each `TransformerEncoderLayer.self_attn` with `need_weights=True` and then plotting heatmaps. 

## Interpreting attention
- Attention tensor shapes: `(B, H, T, S)` = batch, heads, query positions, source positions.
- Rows (query positions) sum to 1 (softmax over source positions).

