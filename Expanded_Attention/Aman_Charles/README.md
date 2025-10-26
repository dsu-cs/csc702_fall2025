# Expanding Attention to Multilingual Contexts

## Project Overview

This project extends the previous *Attention Head Visualization* work by exploring how **Transformer attention and embeddings behave across languages**.
Using a multilingual encoder (**mBERT / XLM-RoBERTa**), we analyze whether semantically similar words in different languages share aligned representations.

The experiment visualizes **token-level cosine similarities**, performs **greedy alignment extraction**, and probes **layer-wise alignment strength** between English and other languages (Spanish, French, German).

---

## Objectives

* Examine how self-attention and token embeddings **transfer across languages**.
* Visualize **cross-lingual alignment** through cosine similarity heatmaps.
* Compare **true translations vs paraphrases** to test semantic sensitivity.
* Identify **which Transformer layers** encode the strongest cross-lingual representations.

---

## Methodology

### **1. Model and Data**

* Model: `bert-base-multilingual-cased` (mBERT).
* Languages: English (EN), Spanish (ES), French (FR), and German (DE).
* Sentences were chosen to be semantically equivalent across languages.

### **2. Steps**

1. **Tokenize** both English and translated sentences.
2. Extract **last-layer contextual embeddings** for each token.
3. Compute **cosine similarity matrices** between EN and foreign tokens.
4. Apply a **greedy alignment algorithm** to pair most similar token embeddings.
5. Visualize similarity heatmaps and analyze alignment pairs.
6. Conduct **layer-wise probing** to measure average EN↔XX cosine similarity across all 12 Transformer layers.

---

## Results & Visualizations

### English ↔ Spanish

**Heatmap Observation:**

* Bright diagonal indicates strong alignment between semantically equivalent tokens.
* Correct matches include:

  * `The ↔ El`
  * `is ↔ está`
  * `on ↔ en`
  * `cat ↔ ga(to)`
* Sentence-level tokens `[CLS] ↔ [CLS]` and `[SEP] ↔ [SEP]` show the highest similarity (~0.95).
* Subword matches (`sleep` ↔ `dur`, `##ing` ↔ `##do`) reveal WordPiece segmentation effects.

**Conclusion:** mBERT effectively learns cross-lingual alignment between English and Spanish, maintaining high semantic coherence.

---

### English ↔ French

**Observation:**

* Consistent diagonal but slightly weaker alignment (~0.45–0.6).
* Key pairs:

  * `animal ↔ animal`
  * `street ↔ rue`
  * `because ↔ parce`
  * `was ↔ était`
* Some `[UNK]` tokens appear due to diacritics not covered by WordPiece vocabulary.

**Conclusion:** Cross-lingual attention alignment is stable but mildly reduced for French due to morphological and accent-based tokenization differences.

---

### English ↔ German

**Observation:**

* Strong token-level alignment:

  * `book ↔ Buch`
  * `put ↔ stellte`
  * `up ↔ auf`
  * `again ↔ wieder`
* High cosine similarities (~0.6–0.7).
* Multi-token German compounds still align meaningfully across subwords.

**Conclusion:** German-English pairs exhibit clear alignment; mBERT generalizes across complex morphology.

---

## Layer-Wise Alignment Probing

**Goal:** Identify which layers of mBERT encode the most cross-lingual similarity.

| Layer Range | Avg EN↔XX Cosine | Interpretation                   |
| ----------- | ---------------- | -------------------------------- |
| Layers 1–3  | 0.2–0.35         | Lexical / language-specific      |
| Layers 4–7  | 0.4–0.6          | Semantic abstraction begins      |
| Layers 8–11 | 0.65–0.7+        | Peak cross-lingual alignment     |
| Layer 12    | ↓ to 0.35        | Language-specific specialization |

**Conclusion:**
Cross-lingual alignment **emerges in mid-layers** and peaks between **layers 9–11**, then decreases slightly as the model becomes more task-tuned at the top.

---

## Interpretation

* mBERT’s shared multilingual embedding space enables **unsupervised translation alignment**.
* Attention and contextual similarity maps show **parallel meaning structures** across languages.
* Word order and syntactic correspondence are naturally captured through self-attention.
* Morphological richness and diacritics slightly weaken subword alignment precision.
* The model’s **mid-to-late layers** are the most universal across languages.

---

## Key Findings

| Aspect                       | Observation                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Semantic Alignment**       | English words align with their translation equivalents across languages (e.g., *book ↔ Buch*, *is ↔ está*). |
| **Sentence-Level Embedding** | `[CLS]` tokens show near-perfect similarity (~0.9–0.95), capturing shared overall meaning.                  |
| **Subword Effect**           | WordPiece segmentation leads to fragment-level matches (e.g., `sleep` ↔ `dur`, `##ing` ↔ `##do`).           |
| **Language Morphology**      | Alignment strength decreases slightly for morphologically rich or accented languages.                       |
| **Cross-Layer Behavior**     | Mid-layers produce the most language-agnostic features.                                                     |

---

## Conclusion

This experiment demonstrates that **Transformer attention and embeddings naturally expand beyond language boundaries**.
Without explicit translation supervision, multilingual BERT learns **shared semantic structures** across English, Spanish, French, and German.

These results show that **expanding attention to multilingual contexts** not only broadens model understanding but also reveals how deeply Transformers internalize universal linguistic patterns.
