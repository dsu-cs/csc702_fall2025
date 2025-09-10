"""
emb_cli.py — Simple CLI for exploring word embeddings.
Usage examples:
  python emb_cli.py --file /path/to/Sample_vectors.txt vocab --top 20
  python emb_cli.py --file /path/to/Sample_vectors.txt neighbors --word company --top 10
  python emb_cli.py --file /path/to/Sample_vectors.txt analogy --a king --b man --c woman --top 5
  python emb_cli.py --file /path/to/Sample_vectors.txt dim
  python emb_cli.py --file /path/to/Sample_vectors.txt pca --words company market million year this --out pca.png
"""
import argparse
import sys
import re
import numpy as np
from pathlib import Path

#matplotlib
def lazyMatplotlib():
    import matplotlib.pyplot as plt
    return plt

def loadEmbeddings(txt_path: Path):
    vocab = []
    vecs = []
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # skip numeric-only stray lines
            if re.fullmatch(r"[-+0-9eE\.\s]+", line):
                continue
            token = parts[0]
            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            if vec.size == 0:
                continue
            vocab.append(token)
            vecs.append(vec)
    if not vecs:
        raise SystemExit("No vectors parsed. Check file format.")
    # Pad/truncate to consistent dimension
    dim = min(v.shape[0] for v in vecs)
    vecs = np.vstack([v[:dim] for v in vecs])
    return vocab, vecs

class EmbIndex:
    def __init__(self, vocab, vecs):
        self.vocab = vocab
        self.vecs = vecs
        self.dim = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        self.unit = vecs / norms
        self.index = {w:i for i,w in enumerate(vocab)}

    def check(self, w):
        if w not in self.index:
            raise KeyError(f"'{w}' not in vocabulary")

    def neighbors(self, word, top=10):
        self.check(word)
        i = self.index[word]
        sims = (self.unit @ self.unit[i:i+1].T).ravel()
        order = np.argsort(-sims)
        out = []
        for j in order:
            if j == i: 
                continue
            out.append((self.vocab[j], float(sims[j])))
            if len(out) >= top:
                break
        return out

    def analogy(self, a, b, c, top=10):
        for w in (a,b,c): self.check(w)
        vec = self.unit[self.index[b]] - self.unit[self.index[a]] + self.unit[self.index[c]]
        sims = (self.unit @ vec.reshape(-1,1)).ravel()
        banned = {self.index[a], self.index[b], self.index[c]}
        order = [j for j in np.argsort(-sims) if j not in banned][:top]
        return [(self.vocab[j], float(sims[j])) for j in order]

    def has(self, word):
        return word in self.index

def cmdVocab(ei: EmbIndex, args):
    top = args.top
    words = ei.vocab[:top] if top else ei.vocab
    for w in words:
        print(w)

def cmdNeighbors(ei: EmbIndex, args):
    out = ei.neighbors(args.word, top=args.top)
    print(f"Nearest to '{args.word}':")
    for w, s in out:
        print(f"  {w:20s}  cos={s:.4f}")

def cmdAnalogy(ei: EmbIndex, args):
    out = ei.analogy(args.a, args.b, args.c, top=args.top)
    print(f"{args.a} : {args.b} :: {args.c} : ?")
    for w, s in out:
        print(f"  {w:20s}  cos={s:.4f}")

def cmdDim(ei: EmbIndex, args):
    print(f"Vocab size: {len(ei.vocab)}") 
    print(f"Vector dim: {ei.dim}")

def cmdPca(ei: EmbIndex, args):
    words = args.words
    missing = [w for w in words if not ei.has(w)]
    if missing:
        print("Warning: missing words:", ", ".join(missing), file=sys.stderr)
    keep = [w for w in words if ei.has(w)]
    if len(keep) < 2:
        raise SystemExit("Need at least 2 known words for PCA plot.")
    X = ei.unit[[ei.index[w] for w in keep]]
    # PCA via SVD
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = (Xc @ Vt[:2].T)
    plt = lazyMatplotlib()
    import matplotlib
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.scatter(coords[:,0], coords[:,1])
    for (x,y), w in zip(coords, keep):
        ax.text(x, y, w)
    ax.set_title("PCA map of selected words")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    out = args.out or "pca.png"
    fig.savefig(out, dpi=200)
    print(f"Saved PCA figure to {out}")

def buildParser():
    p = argparse.ArgumentParser(description="CLI for word embeddings") 
    p.add_argument("--file", required=True, help="Path to embeddings .txt file (word followed by floats per line)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("vocab", help="List first N vocab entries (default all)")
    sp.add_argument("--top", type=int, default=0, help="Show only first N words")
    sp.set_defaults(func=cmdVocab)

    sp = sub.add_parser("neighbors", help="Find nearest neighbors for a word")
    sp.add_argument("--word", required=True)
    sp.add_argument("--top", type=int, default=10)
    sp.set_defaults(func=cmdNeighbors)

    sp = sub.add_parser("analogy", help="Solve a:b::c:? (cosine)")
    sp.add_argument("--a", required=True)
    sp.add_argument("--b", required=True)
    sp.add_argument("--c", required=True)
    sp.add_argument("--top", type=int, default=10)
    sp.set_defaults(func=cmdAnalogy)

    sp = sub.add_parser("dim", help="Show vocab size and vector dimension")
    sp.set_defaults(func=cmdDim)

    sp = sub.add_parser("pca", help="Save a PCA 2D plot for selected words") 
    sp.add_argument("--words", nargs="+", required=True)
    sp.add_argument("--out", default="pca.png")
    sp.set_defaults(func=cmdPca)

    return p

def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = buildParser()
    args = parser.parse_args(argv)

    txt_path = Path(args.file)
    if not txt_path.exists():
        raise SystemExit(f"File not found: {txt_path}")
    vocab, vecs = loadEmbeddings(txt_path)
    ei = EmbIndex(vocab, vecs)

    try:
        args.func(ei, args)
    except KeyError as e:
        raise SystemExit(str(e))

if __name__ == "__main__":
    main()
