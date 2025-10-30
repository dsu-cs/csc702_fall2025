from sentence_transformers import SentenceTransformer
import chromadb
import streamlit as st
from groq import Groq
import os

# disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GROQ_API_KEY"] = ""

# Streamlit page config
st.title("RAG Chatbot")

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize models and database (use @st.cache_resource to avoid reloading)
@st.cache_resource
def initialize_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()
    
    # Try to get existing collection, create if it doesn't exist
    try:
        collection = chroma_client.get_collection(name="docs")
    except:
        collection = chroma_client.create_collection(name="docs")
        # add local text data
        texts = ['RAG combines retrieval and generation.', 'embeddings map text to vector space.']
        embeddings = embedder.encode(texts).tolist()
        collection.add(documents=texts, embeddings=embeddings, ids=['1','2'])
    
    groq_client = Groq()
    return embedder, collection, groq_client

embedder, collection, groq_client = initialize_models()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # RAG retrieval
    query_emb = embedder.encode([prompt]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=2)
    context = '\n'.join(results['documents'][0])
    
    # Build messages with history (only for Groq API)
    conversation_history = []
    for msg in st.session_state.messages[:-1]:  # Exclude the message we just added
        conversation_history.append({
            'role': msg['role'],
            'content': msg['content']
        })
    
    # Add current query with context
    conversation_history.append({
        'role': 'user',
        'content': f'Context: {context}\n\nQuestion: {prompt}'
    })
    
    # Get response from Groq
    response = groq_client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=conversation_history
    )
    
    answer = response.choices[0].message.content
    
    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)