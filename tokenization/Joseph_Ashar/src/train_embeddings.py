import argparse
from pathlib import Path
from gensim.models import Word2Vec, FastText
from .prepare_data import load_corpus, normalize
from .tokenize import word_tokens, char_tokens, train_bpe, load_bpe_model, bpe_tokens, chunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[1] / "data" / "corpus.txt"))
    ap.add_argument("--vector-size", type=int, default=200)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bpe-vocab", type=int, default=8000)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / "results" / "models"
    outdir.mkdir(parents=True, exist_ok=True)

    text = normalize(load_corpus(args.corpus))

    # Word tokens
    w_tokens = word_tokens(text)
    w_seqs = list(chunk(w_tokens, 2000))
    w2v = Word2Vec(sentences=w_seqs, vector_size=args.vector_size, window=args.window,
                   min_count=args.min_count, sg=1, workers=4, epochs=args.epochs)
    w2v.save(str(outdir / "word_w2v.model"))

    ft = FastText(sentences=w_seqs, vector_size=args.vector_size, window=args.window,
                  min_count=args.min_count, sg=1, workers=4, epochs=args.epochs)
    ft.save(str(outdir / "word_fasttext.model"))

    # Char tokens
    c_tokens = char_tokens(text)
    c_seqs = list(chunk(c_tokens, 4000))
    c2v = Word2Vec(sentences=c_seqs, vector_size=args.vector_size, window=args.window,
                   min_count=5, sg=1, workers=4, epochs=args.epochs)
    c2v.save(str(outdir / "char_w2v.model"))

    # BPE tokens
    spm_model = train_bpe(args.corpus, vocab_size=args.bpe_vocab)
    sp = load_bpe_model()
    b_tokens = bpe_tokens(sp, text)
    b_seqs = list(chunk(b_tokens, 2000))
    b2v = Word2Vec(sentences=b_seqs, vector_size=args.vector_size, window=args.window,
                   min_count=5, sg=1, workers=4, epochs=args.epochs)
    b2v.save(str(outdir / "bpe_w2v.model"))

    print("Models saved to:", outdir)

if __name__ == "__main__":
    main()
