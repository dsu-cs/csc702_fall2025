import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from .utils import save_json

def most_similar(model, token, topn=10):
    try:
        return [w for w,_ in model.wv.most_similar(token, topn=topn)]
    except KeyError:
        return []

def simvec(model, token, vocab):
    if token not in model.wv.key_to_index:
        return None
    v = model.wv[token].reshape(1,-1)
    M = np.stack([model.wv[w] for w in vocab])
    return cosine_similarity(v, M).ravel()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "models"))
    ap.add_argument("--anchors", nargs="+", default=["bank","apple","virus","port","model","crash","key","stream","root","cloud","kernel","attack"])
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    mdir = Path(args.models_dir)
    w2v = Word2Vec.load(str(mdir / "word_w2v.model"))
    ft  = FastText.load(str(mdir / "word_fasttext.model"))
    b2v = Word2Vec.load(str(mdir / "bpe_w2v.model"))
    # Char space may not have word anchors; we keep it for optional analysis
    # c2v = Word2Vec.load(str(mdir / "char_w2v.model"))

    # Shared comparison vocab (limit to frequent terms for stability)
    common = [w for w in w2v.wv.index_to_key[:10000] if w in ft.wv.key_to_index]

    rows = []
    neighbor_tables = {}

    for a in args.anchors:
        if a not in w2v.wv.key_to_index or a not in ft.wv.key_to_index:
            continue
        nb_w2v = most_similar(w2v, a, args.topk)
        nb_ft  = most_similar(ft,  a, args.topk)
        jacc_w_ft = len(set(nb_w2v) & set(nb_ft)) / max(1, len(set(nb_w2v) | set(nb_ft)))

        s1 = simvec(w2v, a, common)
        s2 = simvec(ft,  a, common)
        rho_w_ft = float(spearmanr(s1, s2).correlation) if s1 is not None and s2 is not None else float("nan")

        # Word vs BPE (approximate by comparing word tokens present in b2v as whole tokens; may be sparse)
        if a in b2v.wv.key_to_index:
            nb_bpe = most_similar(b2v, a, args.topk)
            jacc_w_bpe = len(set(nb_w2v) & set(nb_bpe)) / max(1, len(set(nb_w2v) | set(nb_bpe)))
        else:
            nb_bpe, jacc_w_bpe = [], 0.0

        rows.append({
            "anchor": a,
            "jaccard@{k}_word_vs_fasttext".format(k=args.topk): jacc_w_ft,
            "spearman_word_vs_fasttext": rho_w_ft,
            "jaccard@{k}_word_vs_bpe".format(k=args.topk): jacc_w_bpe
        })
        neighbor_tables[a] = {"word2vec": nb_w2v, "fasttext": nb_ft, "bpe": nb_bpe}

    df = pd.DataFrame(rows).sort_values("spearman_word_vs_fasttext", ascending=False)
    print(df.to_string(index=False))

    out_metrics = Path(args.models_dir).parents[0] / "metrics.json"
    save_json({"metrics": rows, "neighbors": neighbor_tables}, str(out_metrics))
    print("Saved metrics to", out_metrics)

if __name__ == "__main__":
    main()
