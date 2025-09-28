#Import Libraries
import argparse, os, glob, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#Torch
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:
    TORCH_OK = False

#Load the sample Data files in this case from data folder
def loadTextFolder(data_dir: str):
    texts, labels = [], []
    classes = sorted([p for p in Path(data_dir).glob("*") if p.is_dir()])
    if not classes:
        raise SystemExit(f"No class folders found in {data_dir}. Put files under data/<label>/*.txt")
    for c in classes:
        for f in glob.glob(str(c / "*.txt")):
            try:
                txt = Path(f).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = Path(f).read_text(encoding="latin-1", errors="ignore")
            texts.append(txt)
            labels.append(c.name)
    return texts, np.array(labels)

#Build Tfid Vectors
def buildVectorizers(texts, maxFeatures=30000, ngramRange=(1,2), minDf=2):
    bow = CountVectorizer(max_features=maxFeatures, ngram_range=ngramRange, min_df=minDf)
    tfidf = TfidfVectorizer(max_features=maxFeatures, ngram_range=ngramRange, min_df=minDf, norm="l2")
    XBow = bow.fit_transform(texts)
    XTfidf = tfidf.fit_transform(texts)
    return XBow, XTfidf, bow, tfidf

#lsa Embeddings
def lsaEmbeddings(XTfidf, dims=100, randomState=42):
    dimsEff = safeDim(dims, XTfidf)
    if dimsEff != dims:
        print(f"[LSA] Requested {dims} dims but TF-IDF has {XTfidf.shape[1]} features; using {dimsEff}.")
    svd = TruncatedSVD(n_components=dimsEff, random_state=randomState)
    return svd.fit_transform(XTfidf), svd

#Nfm Embedding
def nmfEmbeddings(XTfidf, dims=100, randomState=42):
    dimsEff = safeDim(dims, XTfidf)
    if dimsEff != dims:
        print(f"[NMF] Requested {dims} dims but TF-IDF has {XTfidf.shape[1]} features; using {dimsEff}.")
    nmf = NMF(n_components=dimsEff, init="nndsvda", random_state=randomState, max_iter=400)
    return nmf.fit_transform(XTfidf), nmf

class AE(nn.Module):
    def __init__(self, inputDim, hidden=512, bottleneck=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputDim, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, inputDim)
        )
    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

#auto encoder embedding
def autoencoderEmbeddings(XTfidf, dims=100, epochs=15, lr=1e-3, batch=64, device="cpu"):
    if not TORCH_OK:
        raise RuntimeError("PyTorch not installed; run with --no-autoencoder or install torch.")
    X = XTfidf.toarray().astype(np.float32)
    inputDim = X.shape[1]
    model = AE(inputDim, hidden=min(1024, max(128, inputDim//4)), bottleneck=dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    tens = torch.from_numpy(X).to(device)
    ds = torch.utils.data.TensorDataset(tens)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            opt.zero_grad()
            xhat, _ = model(xb)
            loss = lossfn(xhat, xb)
            loss.backward()
            opt.step()
    # get embeddings
    model.eval()
    with torch.no_grad():
        _, Z = model(tens)
    return Z.cpu().numpy(), model

def evaluate(name, X, y):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    # smallest class size
    _, counts = np.unique(y, return_counts=True)
    min_per_class = counts.min()
    nSplits = max(2, min(5, int(min_per_class)))

    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=nSplits, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"[{name}] {nSplits}-fold accuracy: mean={acc.mean():.4f}  std={acc.std():.4f}")
    return acc.mean(), acc.std()
#Plot the graph
def plotTsne(Z, y, title="t-SNE"):
    y_uni = sorted(np.unique(y))
    y_map = {c:i for i,c in enumerate(y_uni)}
    y_idx = np.array([y_map[c] for c in y])
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(y)//10)), init="pca", learning_rate="auto", random_state=42)
    pts = tsne.fit_transform(Z)
    plt.figure(figsize=(7,6))
    for cls in y_uni:
        idx = (y == cls)
        plt.scatter(pts[idx,0], pts[idx,1], s=20, alpha=0.7, label=cls)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

#Save CSV files
def saveCsv(path, Z, y, prefix):
    out = Path(path) / f"{prefix}_embeddings.csv"
    header = ",".join([f"d{i}" for i in range(Z.shape[1])]) + ",label\n"
    with open(out, "w", encoding="utf-8") as f:
        f.write(header)
        for row, lab in zip(Z, y):
            f.write(",".join(map(lambda v: f"{v:.6f}", row)) + f",{lab}\n")
    print(f"Saved: {out}")
    
def safeDim(requested, X):
    # components must be <= n_features for TruncatedSVD
    n_samples, n_features = X.shape
    # keep a little headroom; at least 2 dims
    return max(2, min(requested, n_features - 1 if n_features > 2 else n_features))
    
#Input Arguments
def main():
    ap = argparse.ArgumentParser(description="Bag of Words => Embeddings (LSA, NMF, Autoencoder)")
    ap.add_argument("--data-dir", required=True, help="Folder with class subfolders full of .txt files")
    ap.add_argument("--embed-dims", type=int, default=100)
    ap.add_argument("--plot", action="store_true", help="Plot t-SNE for each embedding")
    ap.add_argument("--save", action="store_true", help="Save embeddings CSVs")
    ap.add_argument("--no-autoencoder", action="store_true", help="Skip autoencoder")
    args = ap.parse_args()

    texts, y = loadTextFolder(args.data_dir)
    print(f"Loaded {len(texts)} docs across {len(np.unique(y))} classes.")

    # Vectorize
    XBow, XTfidf, bow, tfidf = buildVectorizers(texts)
    print(f"BoW shape: {XBow.shape} | TF-IDF shape: {XTfidf.shape}")

    # Baseline
    evaluate("TF-IDF (baseline)", XTfidf, y)

    # LSA
    Z_lsa, svd = lsaEmbeddings(XTfidf, dims=args.embed_dims)
    evaluate(f"LSA-{args.embed_dims}", Z_lsa, y)
    if args.plot: plotTsne(Z_lsa, y, f"t-SNE LSA-{args.embed_dims}")
    if args.save: saveCsv(".", Z_lsa, y, f"lsa{args.embed_dims}")

    # NMF
    Z_nmf, nmf_model = nmfEmbeddings(XTfidf, dims=args.embed_dims)
    evaluate(f"NMF-{args.embed_dims}", Z_nmf, y)
    if args.plot: plotTsne(Z_nmf, y, f"t-SNE NMF-{args.embed_dims}")
    if args.save: saveCsv(".", Z_nmf, y, f"nmf{args.embed_dims}")

    # Autoencoder 
    if not args.no_autoencoder:
        if not TORCH_OK:
            print("PyTorch not available; skipping autoencoder.")
        else:
            device = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"
            Z_ae, _ = autoencoderEmbeddings(XTfidf, dims=args.embed_dims, device=device)
            evaluate(f"AE-{args.embed_dims}", Z_ae, y)
            if args.plot: plotTsne(Z_ae, y, f"t-SNE AE-{args.embed_dims}")
            if args.save: saveCsv(".", Z_ae, y, f"ae{args.embed_dims}")

if __name__ == "__main__":
    main()
