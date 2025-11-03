# RAG Application

## Issues we ran into
When first trying to set things up, we attempted to do everything locally on the vm. Our motivation was to use small models that were more likely limited in their knowledge in order to have more noticable results when incorporating extra information/context with RAG. We were using ollama and some of the infrastructure built around that along with Qdrant for our vector database that ran on a docker container. We ran into issues with memory usage, despite utilizing small models. Rather than beating our heads against the wall just trying to get our pipeline set up, we transferred over to utilizing API calls to Groq, mainly because it seemed to be the most straighforward and it didn't sit behind a paywall like several others.

## Alternate approach
We pivoted to our Groq API calls and got a very basic llama 3.1 model running that really wanted to talk about RAG (it only had 2 "documents" which were each a phrase regarding RAG). We then transfered this into two programs that run interactively as chatbots. The `chatbot.py` script is simple and runs in the terminal as a normal python script. The `pretty_chatbot.py` file is run using `streatlit run pretty_chatbot.py` and provides an easy-to-use web interface to interact with the bot through.
**Be sure to insert your Groq API key before running**

## Required Packages
`sentence-transformers`  
`chromadb`  
`groq`  
`streamlit`

---

## Current Workflow (Single Notebook Only)

This repository now runs entirely from a single Jupyter/Colab notebook:

**File:** `rag_project.ipynb`  
You do not need to run any other scripts. The notebook:
- Extracts text from a PDF with `pypdf`
- Chunks and embeds the text using a SentenceTransformer
- Stores vectors in a local ChromaDB collection (`./db`)
- Runs a simple interactive RAG chat loop that queries the collection and calls the Groq LLM (`llama-3.1-8b-instant`)

> Note: The earlier CLI/Streamlit paths are no longer required for this workflow.

---

## Prerequisites

- Python 3.9+ with Jupyter or Google Colab
- Groq API key (environment variable `GROQ_API_KEY`)
- Local filesystem access to a PDF you want to index

---

## Install Dependencies

If running locally (recommended to use a virtual environment):

```bash
pip install sentence-transformers chromadb groq pypdf
````

If you open the notebook in Colab, the first cells already include `pip install` lines for the required packages.

---

## Environment Setup

Set your Groq API key before running the notebook:

macOS / Linux:

```bash
export GROQ_API_KEY="your_key_here"
```

Windows PowerShell:

```powershell
setx GROQ_API_KEY "your_key_here"
```

> Security note: Remove any hard-coded API keys inside the notebook and rely on the environment variable above.

---

## Configure Data Inputs

In the notebook you will see variables like:

```python
pdf_path = "/content/Artificial intelligence - Wikipedia.pdf"   # your file name
# pdf_path = "/content/Retrieval-augmented generation - Wikipedia.pdf"   # alternative
txt_path = "/content/ai_wikipedia.txt"
```

* Replace `pdf_path` with the path to your own PDF.
* `txt_path` is where extracted text will be saved.
* The notebook uses `pypdf` to read the PDF and write out a plain-text file that will be chunked and embedded.

---

## Run the Notebook

1. Open `rag_project.ipynb` in Jupyter or Colab.
2. Run cells **top to bottom**:

   * Install dependencies (if local)
   * Set `GROQ_API_KEY` (or confirm it’s set in your environment)
   * Extract text from your PDF into a `.txt`
   * Ingest that `.txt` into Chroma:

     * Splits into word chunks (defaults: `chunk_size=300`, `overlap=50`)
     * Encodes with the selected embedder
     * Inserts into collection `research` in a persistent Chroma client at `./db`
   * Start the conversation loop:

     * You’ll be prompted in the console cell with `You: `
     * The system will retrieve relevant chunks and call Groq’s `llama-3.1-8b-instant`
     * The response is printed as `Bot: ...`

No additional scripts are required.

---

## Model Settings

* **Vector DB**: ChromaDB persistent client at `./db`
* **Collection**: `research`
* **LLM**: Groq model `llama-3.1-8b-instant`
* **Default prompt style**: System prompt guides the LLM to use retrieved context

---

## Embedding Model Quality Note

We tested two SentenceTransformer models for retrieval:

```python
# Initial (poorer results):
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
```

With `all-MiniLM-L6-v2`, the RAG retrieval and final answers were often not aligned with the query.

```python
# Updated (better results):
embedder = SentenceTransformer("msmarco-distilbert-base-v4")
```

After switching to `msmarco-distilbert-base-v4`, retrieval accuracy and answer quality improved noticeably. Model choice is critical for RAG performance, especially with small corpora.

---

## Project Structure (minimal)

```
.
├── rag_project.ipynb      # The only file you need to run
├── db/                    # Created by Chroma (persistence)
└── README.md
```

> The `db/` directory is created automatically when the notebook runs and persists your vector store locally.

---

## Troubleshooting

* **No responses / API errors**: Ensure `GROQ_API_KEY` is set in your environment before starting the conversation loop.
* **File not found**: Check `pdf_path` points to a real file in your environment (Colab vs local differs).
* **Poor retrieval quality**: Confirm you are using `msmarco-distilbert-base-v4`, and adjust chunk size/overlap as needed.
* **Permission issues on `./db`**: Ensure the process has write access to the project directory.

---
