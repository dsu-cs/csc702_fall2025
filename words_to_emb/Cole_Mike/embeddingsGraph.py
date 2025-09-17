from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [simple_preprocess(line) for line in lines]

def train_model(sentences):
    return Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4, sg=1)

def find_top_matches(model, top_n_words=10, top_n_matches=5):
    top_words = model.wv.index_to_key[:top_n_words]
    results = {}
    for word in top_words:
        try:
            matches = model.wv.most_similar(word, topn=top_n_matches)
            results[word] = matches
        except KeyError:
            results[word] = []
    return results

def print_matches(title, matches):
    print(f"\n📘 {title}")
    for word, similar_list in matches.items():
        print(f"\n🔹 '{word}' → Top matches:")
        for match, score in similar_list:
            print(f"   {match}: {score:.3f}")

def plot_embeddings(model, title):
    words = list(model.wv.index_to_key)
    vectors = model.wv[words]
    top_words = model.wv.index_to_key[:100]

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    vectors_2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for (x, y), word in zip(vectors_2d, top_words):
        plt.scatter(x, y)
        plt.text(x + 0.02, y + 0.02, word, fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file1 = "LesMiserables_English.txt"
    file2 = "LosMiserables_Spanish.txt"

    data1 = load_data(file1)
    data2 = load_data(file2)

    model1 = train_model(data1)
    model2 = train_model(data2)

    matches1 = find_top_matches(model1, top_n_words=10, top_n_matches=5)
    matches2 = find_top_matches(model2, top_n_words=10, top_n_matches=5)

    print_matches("English File Top Word Matches", matches1)
    print_matches("Spanish Top Word Matches", matches2)
    
    plot_embeddings(model1, "Top 100 Word Embeddings from English File")
    plot_embeddings(model2, "Top 100 Word Embeddings from Spanish File")
