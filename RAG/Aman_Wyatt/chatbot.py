from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import os

# disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GROQ_API_KEY"] = ""

# initialize embedding model & vector db
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
# create or pull collection
collection = chroma_client.create_collection(name="docs")
# collection = client.get_collection(name="docs")

# add local text data
texts = ['RAG combines retrieval and generation.', 'embeddings map text to vector space.']
embeddings = embedder.encode(texts).tolist()
collection.add(documents=texts, embeddings=embeddings, ids=['1','2'])

# initialize groq client
groq_client = Groq()

# conversation loop
conversation_history = []
while True:
    query = input("\nYou: ")
    if query.lower() in ['exit', 'quit', 'q']:
        break

    # rag retrieval
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=2)
    context = '\n'.join(results['documents'][0])

    # build messages and include history
    messages = conversation_history.copy()
    messages.append({
        'role': 'user',
        'content': f'Context: {context}\n\nQuestion: {query}'
    })

    response = groq_client.chat.completions.create(
        model = 'llama-3.1-8b-instant',
        messages = messages
    )

    answer = response.choices[0].message.content
    print(f'Bot: {answer}')

    # update history
    conversation_history.append({'role':'user', 'content':'query'})
    conversation_history.append({'role':'assistant', 'content':answer})
