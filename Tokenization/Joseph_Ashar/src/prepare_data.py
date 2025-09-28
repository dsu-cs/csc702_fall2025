from pathlib import Path
import re

def load_corpus(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus not found at {p}. Put a UTF-8 text file at data/corpus.txt")
    return p.read_text(encoding="utf-8", errors="ignore")

def normalize(text: str) -> str:
    # Basic normalization; keep letters, digits, basic punctuation and whitespace
    # Lowercase to simplify vocabulary
    text = text.lower()
    # Replace fancy quotes/dashes
    text = text.replace("“","\"").replace("”","\"").replace("’","'").replace("–","-").replace("—","-")
    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text)
    return text
