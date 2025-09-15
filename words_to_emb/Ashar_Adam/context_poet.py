"""
Shakespeare Context‑Poet
------------------------
A small "AI-flavored" Python app that:
  1) Downloads Shakespeare (Project Gutenberg #100)
  2) Finds 10 random words that appear in **diverse contexts**
  3) Displays those words with several context snippets **in a box** (GUI)
  4) Generates a fresh poem that uses each word **honoring its contexts**

Run:
  python context_poet.py

Notes:
- Uses only stdlib + requests (for download). No heavy NLP deps.
- "Different contexts" ≈ the word appears with sufficiently varied neighbors.
- The poem generator blends context keywords to keep each word's flavor.
"""
from __future__ import annotations
import random
import re
import textwrap
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import requests

# ---------------------------
# Config
# ---------------------------
SEED = 42
random.seed(SEED)
MAX_LEN = 1_000_000  # min size for a valid cached download
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True, parents=True)
RAW_PATH = DATA_DIR / "shakespeare.txt"

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
    "https://www.gutenberg.org/files/100/100-0.txt",
    "https://www.gutenberg.org/ebooks/100.txt.utf-8",
]

HEADER_RE = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
FOOTER_RE = re.compile(r"\*\*\*\s*END OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_RE = re.compile(r"[a-zA-Z']+")

# ---------------------------
# Data
# ---------------------------

def download_text() -> str:
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > MAX_LEN:
        return RAW_PATH.read_text(encoding="utf-8", errors="ignore")
    last = None
    for url in GUTENBERG_URLS:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            RAW_PATH.write_text(r.text, encoding="utf-8")
            return r.text
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to download Shakespeare: {last}")


def strip_boilerplate(text: str) -> str:
    start = HEADER_RE.search(text)
    end = FOOTER_RE.search(text)
    if start and end:
        core = text[start.end():end.start()]
    else:
        core = text
    core = core.replace("\r", "\n")
    core = re.sub(r"\n{3,}", "\n\n", core)
    return core


def to_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def tokenize(s: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(s)]

# ---------------------------
# Context mining
# ---------------------------

def build_index(sentences: list[str]):
    # word -> list[ (sentence, left_neighbor, right_neighbor) ]
    idx = {}
    for s in sentences:
        toks = tokenize(s)
        for i, w in enumerate(toks):
            left = toks[i-1] if i > 0 else ""
            right = toks[i+1] if i+1 < len(toks) else ""
            idx.setdefault(w, []).append((s, left, right))
    return idx


def context_diversity(entries: list[tuple[str, str, str]]) -> float:
    # Diversity via unique left/right bigrams around the word
    sigs = set()
    for _, l, r in entries:
        if l or r:
            sigs.add((l, r))
    return len(sigs)


def pick_words(idx, min_freq=20, min_diversity=10, k=10) -> list[str]:
    candidates = [w for w, lst in idx.items() if len(lst) >= min_freq and w.isalpha() and len(w) > 2]
    random.shuffle(candidates)
    chosen = []
    for w in candidates:
        if context_diversity(idx[w]) >= min_diversity:
            chosen.append(w)
        if len(chosen) >= k:
            break
    # fallback if not enough found
    if len(chosen) < k:
        extra = [w for w in candidates if w not in chosen]
        chosen += extra[:k-len(chosen)]
    return chosen[:k]


def sample_contexts(entries: list[tuple[str, str, str]], n=3) -> list[str]:
    # Prefer distinct neighbor signatures; truncate sentence around the word
    by_sig = {}
    for s, l, r in entries:
        sig = (l, r)
        by_sig.setdefault(sig, []).append(s)
    # pick up to n diverse sigs
    keys = list(by_sig.keys())
    random.shuffle(keys)
    snippets = []
    for sig in keys[: n*3]:  # widen pool
        s = random.choice(by_sig[sig])
        # compact snippet
        snippet = re.sub(r"\s+", " ", s).strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "…"
        snippets.append(snippet)
        if len(snippets) >= n:
            break
    if not snippets:
        snippets = [re.sub(r"\s+", " ", entries[0][0]).strip()[:160]]
    return snippets

# ---------------------------
# Poem generator
# ---------------------------
TEMPLATES = [
    "O {w}, where {a} leans to {b}, thou changest hue.",
    "In {a}'s hush and {b}'s thunder, {w} walks anew.",
    "They named thee {w}, yet in {a} and {b} thou art two.",
    "Between {a} and {b}, {w} finds a secret view.",
    "When {a} fades and {b} blooms, {w} breaks through.",
]


def extract_context_tags(snippets: list[str], w: str, top=2) -> list[str]:
    # From snippets, pick nontrivial words near target to use as tags
    counts = {}
    stop = set("the and or but for nor so yet a an to of in on by with not be is are was were hath doth did from thou thee thy ye this that these those here there our your their his her its o oh".split())
    for s in snippets:
        toks = tokenize(s)
        for i, tok in enumerate(toks):
            if tok == w:
                # neighbors window
                win = toks[max(0, i-4): i] + toks[i+1: i+5]
                for t in win:
                    if t.isalpha() and t not in stop and len(t) > 2:
                        counts[t] = counts.get(t, 0) + 1
    # fallback: all tokens
    if not counts:
        for s in snippets:
            for t in tokenize(s):
                if t.isalpha() and t not in stop and len(t) > 3 and t != w:
                    counts[t] = counts.get(t, 0) + 1
    tags = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:top]
    # Ensure at least two tags
    while len(tags) < top:
        tags.append(random.choice(["night", "light", "storm", "dream", "shadow", "rose"]))
    return tags[:top]


def compose_poem(selection: dict[str, list[str]]) -> str:
    # Build stanzas; each selected word contributes a line using its tags
    lines = []
    opening = "\n".join([
        "A Poem of Many Meanings",
        "by A Little Automaton",
        "",
    ])
    lines.append(opening)
    for i, (w, snippets) in enumerate(selection.items(), 1):
        a, b = extract_context_tags(snippets, w, top=2)
        template = random.choice(TEMPLATES)
        line = template.format(w=w, a=a, b=b)
        lines.append(f"{i:02d}. {line}")
    closing = "\n".join(["", "— fin —"]) 
    lines.append(closing)
    return "\n".join(lines)

# ---------------------------
# GUI
# ---------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Shakespeare Context‑Poet")
        self.geometry("920x700")
        self.minsize(820, 620)
        self.configure(padx=10, pady=10)

        # Controls
        ctl = ttk.Frame(self)
        ctl.pack(fill=tk.X)
        self.btn_pick = ttk.Button(ctl, text="Pick 10 Words", command=self.pick_words)
        self.btn_pick.pack(side=tk.LEFT)
        self.btn_poem = ttk.Button(ctl, text="Write Poem", command=self.write_poem, state=tk.DISABLED)
        self.btn_poem.pack(side=tk.LEFT, padx=8)

        # Context box
        self.context_box = tk.Text(self, wrap=tk.WORD, height=18, bd=2, relief=tk.GROOVE)
        self.context_box.pack(fill=tk.BOTH, expand=True, pady=(10, 8))
        self.context_box.insert(tk.END, "Click ‘Pick 10 Words’ to begin…\n")

        # Poem box
        self.poem_box = tk.Text(self, wrap=tk.WORD, height=14, bd=2, relief=tk.GROOVE)
        self.poem_box.pack(fill=tk.BOTH, expand=True)

        # Status
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor=tk.W).pack(fill=tk.X, pady=(6,0))

        self.sentences = []
        self.index = None
        self.selection = {}  # word -> [snippets]

        # preload in background
        threading.Thread(target=self._load_corpus, daemon=True).start()

    def _load_corpus(self):
        try:
            self.status.set("Downloading Shakespeare…")
            raw = download_text()
            self.status.set("Preparing text…")
            core = strip_boilerplate(raw)
            self.sentences = to_sentences(core)
            self.index = build_index(self.sentences)
            self.status.set("Corpus ready. Pick words!")
        except Exception as e:
            self.status.set(f"Error: {e}")

    def pick_words(self):
        if not self.index:
            self.status.set("Still loading corpus. Please wait a moment…")
            return
        words = pick_words(self.index, min_freq=25, min_diversity=12, k=10)
        self.selection = {}
        self.context_box.delete("1.0", tk.END)
        for w in words:
            entries = self.index[w]
            snippets = sample_contexts(entries, n=3)
            self.selection[w] = snippets
            self.context_box.insert(tk.END, f"\n— {w.upper()} —\n",)
            for s in snippets:
                self.context_box.insert(tk.END, f"  · {s}\n")
        self.context_box.see(tk.END)
        self.status.set("Selected 10 words with diverse contexts.")
        self.btn_poem.config(state=tk.NORMAL)

    def write_poem(self):
        if not self.selection:
            self.status.set("Pick words first.")
            return
        poem = compose_poem(self.selection)
        self.poem_box.delete("1.0", tk.END)
        self.poem_box.insert(tk.END, poem)
        self.poem_box.see(tk.END)
        self.status.set("Poem composed.")


if __name__ == "__main__":
    App().mainloop()

