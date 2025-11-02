 Quantum Computing Chatbot.
 This project builds a local Retrieval Augmented Generation chatbot that answers questions about quantum computing using Wikipedia data. It downloads topics, cleans them, splits them into chunks, creates embeddings, stores them in a vector database, and uses a local LLM with Ollama to answer questions.


It collects articles on topics like quantum computing, qubits, entanglement, Shor's algorithm, and more. Each article is saved, cleaned to remove citations, HTML, and extra formatting, and then split into smaller text chunks.

These chunks are converted into embeddings using the HuggingFace MiniLM model and stored in a local Chroma vector database. When a user asks a question, the system searches for the most relevant chunks, sends them to a local language model from Ollama (such as llama3), and generates an answer based only on that information. It also shows which files the answer came from.

The chatbot works offline after setup and can remember previous questions in the same session, making it feel more like a real conversation. This project is useful for learning and exploring quantum computing with verified sources and no internet dependency.


 Features
 Downloads quantum computing topics from Wikipedia
 Cleans and prepares the dataset 
 Splits text into chunks for retrieval
 Creates embeddings using HuggingFace models
 Stores embeddings in a Chroma vector database
 Answers questions using a local Ollama model 3.2
 Shows document sources used for each answer
 Supports basic conversation memory

 Installation
pip install langchain langchain-community langchain-core langchain-ollama
pip install chromadb sentence-transformers wikipedia-api
pip install ollama

download locally
sudo snap install ollama
ollama pull llama3.2
