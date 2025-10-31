# 🧠 Hybrid Sequential Transformer for Sentiment Analysis

This project implements a **hybrid Transformer architecture** that combines a custom-built **Mini-Transformer** (with dot-product self-attention) and **BERT** for sentiment classification on the **IMDB movie review dataset**. The model is designed to explore and compare different attention mechanisms and their effectiveness in NLP.

---

## 🚀 Features
- Custom **Mini-Transformer Encoder** implementation from scratch using PyTorch.
- **BERT integration** for hybrid sequential learning.
- **Attention visualization** across layers and heads.
- **Performance comparison** between Mini-Transformer, BERT, and Hybrid attention.
- Modular training, evaluation, and saving pipeline.

---

## 🧩 Project Structure

| Section | Description |
|----------|--------------|
| **1. Environment Setup** | Installs and imports required libraries. |
| **2. Data Loading & Preprocessing** | Loads IMDB dataset, tokenizes text, and builds vocabulary. |
| **3. Model Definition** | Implements Mini-Transformer, positional encoding, and HybridSequentialClassifier (Mini + BERT). |
| **4. Training Loop** | Trains the hybrid model with evaluation per epoch. |
| **5. Evaluation** | Computes accuracy for different attention combinations (Mini-only, BERT-only, Hybrid). |
| **6. Attention Visualization** | Extracts and visualizes self-attention weights layer-wise. |
| **7. Model Saving** | Saves trained model weights and vocabulary mappings. |

---

## ⚙️ Requirements
```bash
pip install torch torchvision transformers pandas scikit-learn matplotlib seaborn
```

---

## 🧮 Training

Run all cells sequentially in **Google Colab** or **Jupyter Notebook**:
```python
# Train and evaluate
python Transfomer.ipynb
```

You can adjust:
- `freeze_bert=True` → freeze BERT weights  
- `EPOCHS`, `LR`, and `MAX_LEN` for faster/larger training

---

## 📊 Results
- Achieves ~87% test accuracy on IMDB dataset.
- Demonstrates consistent performance across attention mechanisms.
- Visualization shows clear focus of self-attention heads on sentiment-heavy tokens.

---

## 💾 Outputs
After successful training:
- `hybrid_sentiment_model.pt` → saved model weights  
- `vocab.json` → vocabulary mappings  

---

## 🧠 Visualization Example
Attention maps and sentiment class predictions can be visualized directly in the notebook through matplotlib and seaborn heatmaps.

---

## 🧍 Author
**[Your Name]**  
Hybrid Transformer implementation for CSC702 (Fall 2025) — Expanded Attention Project.
