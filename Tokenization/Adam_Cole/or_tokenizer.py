import httpx
from itertools import chain
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GUTENBERG_AU_BASE_URL = "https://gutenberg.net.au/"
books = {
    "Down and Out in Paris and London": "/ebooks01/0100171.txt",
    "Burmese Days": "/ebooks02/0200051.txt",
    "A Clergyman's Daughter": "/ebooks02/0200011.txt",
    "Keep the Aspidistra Flying": "/ebooks02/0200021.txt",
    "The Road to Wigan Pier": "/ebooks02/0200391.txt",
    "Homage to Catalonia": "/ebooks02/0201111.txt",
    "Coming up for Air": "/ebooks02/0200031.txt",
    "Animal Farm": "/ebooks01/0100011.txt",
    "Nineteen eighty-four": "/ebooks01/0100021.txt"
}

# Get books
DATA.mkdir(parents=True, exist_ok=True)
for title, url in books.items():
    path = DATA / f"{title.lower().replace(' ', '_')}.txt"
    if path.exists():
        print(f"[data] {title} already exists, skipping download.")
        continue
    print(f"[data] Downloading {title} from {GUTENBERG_AU_BASE_URL + url}")
    response = httpx.get(GUTENBERG_AU_BASE_URL + url)
    response.raise_for_status()
    text = response.text

    text = text.split("Author:     George Orwell\n\n")
    if len(text) < 2:
        text = text[0].split("Author: George Orwell\n\n")
    if len(text) < 2:
        text = text[0].split("Author:     George Orwell (pseudonym of Eric Blair) (1903-1950)\n\n")
    
    text = text[1]
    
    text = text.split("""

THE END

""")[0].strip()
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


if not Path("saved_tokenizers/orwell-vocab.json").exists():
    tokenizer = Tokenizer(BPE())
    
    files = []
    for file in DATA.glob("*.txt"):
        files.append(open(file, "r", encoding="utf-8"))
    tokenizer.train_from_iterator(chain(*files))

    Path("saved_tokenizers").mkdir(parents=True, exist_ok=True)

    tokenizer.save("saved_tokenizers/orwell-vocab.json")
else:
    tokenizer = Tokenizer.from_file("saved_tokenizers/orwell-vocab.json")


# Not common enough to have its own token, but "goth" is.
enc = tokenizer.encode("Shoggoth")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")

# Common enough to have its own token.
enc = tokenizer.encode("Cthulhu")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")


enc = tokenizer.encode("It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")

# Something slightly longer, from one of H.P.'s penpals.
enc = tokenizer.encode("Hither came Conan, the Cimmerian, black-haired, sullen-eyed, sword in hand, a thief, a reaver, a slayer, with gigantic melancholies and gigantic mirth, to tread the jeweled thrones of the Earth under his sandalled feet")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")

# Longer. Lovecraft (_The Cats of Ulthar_)
enc = tokenizer.encode("""It is said that in Ulthar, which lies beyond the river Skai, no man may kill a cat; and this I can 
verily believe as I gaze upon him who sitteth purring before the fire. For the cat is cryptic, and 
close to strange things which men cannot see. He is the soul of antique Aegyptus, and bearer 
of tales from forgotten cities in Meroé and Ophir. He is the kin of the jungle’s lords, and heir to 
the secrets of hoary and sinister Africa. The Sphinx is his cousin, and he speaks her 
language; but he is more ancient than the Sphinx, and remembers that which she hath 
forgotten.""")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")