from pathlib import Path
import re
from typing import List, Iterable
import sentencepiece as spm

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")

def word_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text)

def char_tokens(text: str) -> List[str]:
    return list(text)

# ---- BPE via SentencePiece ----
def _sp_model_paths(root: Path):
    spm_dir = root / "results" / "spm"
    spm_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(spm_dir / "bpe")
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    return model_prefix, model_file, vocab_file

def train_bpe(corpus_path: str, vocab_size: int = 8000, character_coverage: float = 1.0, model_type: str = "bpe") -> str:
    root = Path(__file__).resolve().parents[1]
    model_prefix, model_file, _ = _sp_model_paths(root)
    if Path(model_file).exists():
        return model_file
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type
    )
    return model_file

def load_bpe_model() -> spm.SentencePieceProcessor:
    root = Path(__file__).resolve().parents[1]
    _, model_file, _ = _sp_model_paths(root)
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp

def bpe_tokens(sp: spm.SentencePieceProcessor, text: str) -> List[str]:
    return sp.encode_as_pieces(text)

# Chunk the long token list into "sentences" for gensim training
def chunk(tokens: List[str], length: int = 2000) -> Iterable[List[str]]:
    for i in range(0, len(tokens), length):
        yield tokens[i:i+length]
