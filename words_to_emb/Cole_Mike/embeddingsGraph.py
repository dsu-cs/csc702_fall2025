from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("reading file")
with open("theOdyssey.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()


sentences = [line.lower().strip().split() for line in lines if line.strip()]

model = Word2Vec(sentences, vector_size=50, window=3,min_count=1,sg=1,epochs=500)

words = list(model.wv.index_to_key)
vectors = model.wv[words]

tsne = TSNE(n_components=2, random_state=42, perplexity=5)

vectors_2d = tsne.fit_transform(vectors)

plt.figure(figsize=(8,6))

for (x,y), word in zip(vectors_2d, words):
    plt.scatter(x,y)
    plt.text(x + 0.02, y + 0.02, word, fontsize=12)
plt.title("Word Embeddings as 2D")
plt.show()