import httpx
from pathlib import Path

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