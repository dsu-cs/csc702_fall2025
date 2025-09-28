import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, FastText
from sklearn.manifold import TSNE
import umap

def _subset_vectors(model, tokens):
    vecs, keep = [], []
    for t in tokens:
        if t in model.wv.key_to_index:
            vecs.append(model.wv[t])
            keep.append(t)
    return np.stack(vecs), keep

def plot_umap(model, tokens, title, outpath):
    X, labels = _subset_vectors(model, tokens)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0)
    Z = reducer.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(Z[:,0], Z[:,1])
    for i,lab in enumerate(labels):
        plt.text(Z[i,0], Z[i,1], lab, fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_tsne(model, tokens, title, outpath):
    X, labels = _subset_vectors(model, tokens)
    Z = TSNE(n_components=2, perplexity=30, learning_rate=200, init="random", random_state=0, metric="cosine").fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(Z[:,0], Z[:,1])
    for i,lab in enumerate(labels):
        plt.text(Z[i,0], Z[i,1], lab, fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "models"))
    ap.add_argument("--anchors", nargs="+", default=["bank","apple","virus","port","model","crash","key","stream","root","cloud","kernel","attack"])
    args = ap.parse_args()

    mdir = Path(args.models_dir)
    w2v = Word2Vec.load(str(mdir / "word_w2v.model"))
    ft  = FastText.load(str(mdir / "word_fasttext.model"))

    # Build a small token set: anchors + nearest neighbors in Word2Vec
    tokens = set(args.anchors)
    for a in args.anchors:
        if a in w2v.wv.key_to_index:
            for w,_ in w2v.wv.most_similar(a, topn=7):
                tokens.add(w)
    tokens = list(tokens)[:200]

    outdir = mdir.parents[1] / "results" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_umap(w2v, tokens, "UMAP - Word2Vec space (anchors mixed in)", str(outdir / "umap_word2vec.png"))
    plot_umap(ft,  tokens, "UMAP - FastText space (anchors mixed in)", str(outdir / "umap_fasttext.png"))
    plot_tsne(w2v, tokens, "t-SNE - Word2Vec space", str(outdir / "tsne_word2vec.png"))
    plot_tsne(ft,  tokens, "t-SNE - FastText space", str(outdir / "tsne_fasttext.png"))

    print("Saved figures to", outdir)

if __name__ == "__main__":
    main()
