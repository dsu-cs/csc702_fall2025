---

# Local RAG System with LlamaIndex, Ollama, and Wikipedia

This project demonstrates how to build a fully local Retrieval-Augmented Generation (RAG) pipeline using:

* LlamaIndex for document ingestion and indexing
* Ollama for running a local large language model
* Wikipedia API for obtaining source content
* SentenceTransformer embeddings for optional retrieval evaluation

The notebook implements the complete RAG workflow and includes an additional section for evaluating the effect of different chunk sizes on retrieval accuracy.

---

## Project Workflow

1. Retrieve Wikipedia pages using `wikipediaapi`
2. Save pages as `.txt` documents in a `data/` directory
3. Load the documents using `SimpleDirectoryReader` (LlamaIndex)
4. Build a vector-based retrieval index
5. Query the index and generate answers using Ollama
6. Test different chunk sizes to measure retrieval accuracy

---

## Requirements

### Python Libraries

Install the required Python dependencies:

```bash
pip install llama-index-core wikipedia-api sentence-transformers
```

### Ollama Installation

Download and install Ollama from:
[https://ollama.com/download](https://ollama.com/download)

Pull a model:

```bash
ollama pull llama2
```

Start the service:

```bash
ollama serve
```

---

## Directory Structure

```
project/
│── model.ipynb               # Main notebook
│── data/                     # Folder for downloaded Wikipedia pages (auto-created)
└── README.md
```

---

## How to Use

1. Open `model.ipynb` in Jupyter Notebook or VS Code
2. Run the cells in order

   * Wikipedia data collection
   * Document storage
   * Index creation
   * Query execution
   * Chunk evaluation section
3. Modify the topic list in the notebook to customize the Wikipedia pages downloaded
4. Use the RAG pipeline to ask questions about the downloaded content

---

## Chunk Size Evaluation

The notebook includes a simple evaluation script to test retrieval accuracy for different chunk sizes (for example: 128, 256, 512 tokens).
For each chunk size, the script:

* Splits documents into chunks
* Embeds chunks using `all-MiniLM-L6-v2`
* Computes cosine similarity between query embeddings and chunk embeddings
* Checks whether the top retrieved chunk belongs to the correct document
* Reports accuracy for each chunk size

The goal is to help determine an appropriate chunk size for improving information retrieval performance.

---

## Key Concepts Demonstrated

* Local LLM execution
* Offline knowledge retrieval system
* Document chunking and vector search
* Basic evaluation of chunk size effects on retrieval accuracy

---

## Future Improvements

* Add BM25 or hybrid retrieval comparison
* Evaluate recall@k, MRR, and NDCG instead of only accuracy
* Persist vector index to disk
* Support additional local models via Ollama
* Build a lightweight API interface for querying the RAG pipeline

---
