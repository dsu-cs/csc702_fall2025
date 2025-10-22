
---

# Transformer Attention Head Visualization Project

## Overview

This project provides an **interactive and analytical exploration of the internal workings of Transformer models**, specifically focusing on **attention heads** in BERT. It helps visualize how each token in a sentence attends to other tokens, demonstrating how contextual relationships are formed within language models.

You’ll not only *see* how Transformers think but also experiment with their internal mechanics by ablating specific attention heads and observing how predictions change.

---

## Objectives

* Understand **self-attention** in Transformer architectures.
* Visualize and interpret **multi head attention patterns**.
* Compare early vs late layer behaviors.
* Perform **head ablation** to identify important attention heads.
* Observe how **token embeddings** evolve across layers.

---

## Project Structure

| Section     | Description                                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------------------------------ |
| **Cell 1**  | Installs required dependencies (`transformers`, `torch`, `bertviz`, `matplotlib`, etc.).                           |
| **Cell 2**  | Imports libraries and loads pretrained BERT models.                                                                |
| **Cell 3**  | Implements a robust show_heads() helper to handle different BertViz versions.                                    |
| **Cell 4**  | Extracts tokens and attention tensors from the model.                                                              |
| **Cell 5**  | Visualizes attention as static heatmaps for specific layers/heads.                                                 |
| **Cell 6**  | Compares early vs late layers by averaging attention across heads.                                                 |
| **Cell 7**  | Extracts and displays the top tokens that a specific token (e.g., “it”) attends to.                                |
| **Cell 8**  | Performs head ablation experiments on a Masked Language Model to study how removing heads affects predictions. |
| **Cell 9**  | Runs comparative experiments on sentences that differ slightly in meaning (A/B probing).                           |
| **Cell 10** | Visualizes token embeddings using PCA to explore geometric relationships.                               |

---

## Concept Summary

### What is Self-Attention?

Self-attention allows the model to look at **all other words** in a sentence to determine which are most relevant when encoding a particular token. It uses:

* **Queries (Q)**, **Keys (K)**, and **Values (V)** matrices.
* Attention scores computed as softmax(QKᵀ / √dₖ).

### Multi-Head Attention

Each head in a Transformer learns to focus on different linguistic patterns:

* Some track **syntax** (subject → verb).
* Others capture **semantics** (pronoun → antecedent).
* Some specialize in **punctuation**, **names**, or **negations**.

---

## Key Functions

### show_heads(sentence)

Displays an interactive visualization of all attention heads and layers using **BertViz**.

* Hovering over a token highlights where it attends.
* Different heads/layers can be selected to explore varying behaviors.

### show_heatmap(attn, tokens)

Creates a static attention heatmap for a specific layer/head.
Useful for reports or non-interactive environments.

### top_attended_tokens()

Prints the top-k tokens that a given word (like “it”) attends to — revealing contextual understanding.

### make_head_mask() + topk_at_mask()

Implements **head ablation** — removing one attention head and checking how the model’s [MASK] predictions change.
This helps identify which heads are critical for resolving pronouns or maintaining context.

---

## Experiments & Observations

### 1. Attention Visualization

* Early layers show **local attention** (to adjacent words).
* Later layers form **semantic links** (e.g., “it” attending to “animal”).
* Some heads track grammar boundaries or punctuation.

### 2. Layer Comparison

* **Layer 1 (avg heads):** attention mostly uniform or local.
* **Layer 12 (avg heads):** stronger focused connections reflecting meaning and dependency.

### 3. Token-Specific Probing

Changing the sentence:

> “The animal didn’t cross the street because it was too tired.”
> vs
> “The animal didn’t cross the street because it was too wide.”

→ shifts the attention of “it” from **“animal”** → **“street”** in later layers, showing **contextual reasoning**.

### 4. Head Ablation

* Disabling certain heads can slightly alter [MASK] predictions.
* Some heads are redundant, but others are crucial for maintaining correct contextual understanding.

### 5. Embedding Geometry

Early-layer embeddings cluster by surface similarity, while later-layer embeddings cluster semantically — showing the **transition from lexical to conceptual representations**.

---

## Example Commands

In Colab:

```python
show_heads("The animal didn't cross the street because it was too tired.")
```

Generate heatmaps:

```python
show_heatmap(attentions[-1][0,0].cpu().numpy(), tokens, "Last Layer, Head 1")
```

Ablate heads:

```python
hm = make_head_mask(12, 12, drop=(11, 0))  
topk_at_mask(masked_sentence, head_mask=hm)
```

---

## Technologies Used

* **Python 3.10+**
* **PyTorch**
* **Hugging Face Transformers**
* **BertViz** (for interactive visualization)
* **Matplotlib / PCA** (for plots & embedding projections)

---

## Results Summary

| Experiment           | Observation                                     |
| -------------------- | ----------------------------------------------- |
| Attention heatmaps   | Show token-to-token dependencies clearly        |
| Early vs late layers | Early = syntactic/local, Late = semantic/global |
| Token probing        | “it” changes target depending on context        |
| Head ablation        | Reveals redundancy and head specialization      |
| Embedding PCA        | Late layers form semantic clusters              |

---

## Future Work

* Extend visualization to **cross-attention** (encoder–decoder models like BART).
* Apply on **GPT-2** or **T5** for causal/seq2seq analysis.
* Integrate **SHAP** or **Integrated Gradients** for token-level interpretability.
* Create a Streamlit dashboard for interactive visualization.

---

## References


* BertViz documentation: [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)
* Hugging Face Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

---
