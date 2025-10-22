# mini_transformer (IMDB Sentiment)

 contains PyTorch Transformer classifier trained on the IMDB movie-review dataset. It is designed for experiments, quick iteration, and attention-visualization.

## Contents

- `Transfomer.ipynb` - Jupyter notebook with the full pipeline: data loading, tokenization, vocabulary building, dataset/dataloader, model definition, training loop, evaluation, and attention capture/visualization cells.
- `data/IMDB Dataset.csv` 

## Requirements

 Python environment (recommended Python 3.8+).

```powershell
py -3.13 -m pip install torch torchvision pandas scikit-learn matplotlib seaborn optuna datasets
```

## Run training from notebook

- Set `SUBSET = None` to use full dataset.
- Adjust hyperparameters near top (BATCH_SIZE, EPOCHS, LR).

## Visualizing self-attention
The notebook contains cells that capture attention weights.