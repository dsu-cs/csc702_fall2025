# load like 70% of lovecraft and all of orwell's novels. Some of Chesterton
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def get_lovecraft_text(force_download=False):
    TRAIN_PATH = DATA / "lovecraft_train.txt"
    TEST_PATH = DATA / "lovecraft_test.txt"
    if TRAIN_PATH.exists() and TEST_PATH.exists() and not force_download:
        print(f"[data] Loading existing text from {TRAIN_PATH}")
        return (TRAIN_PATH.read_text(encoding="utf-8"), TEST_PATH.read_text(encoding="utf-8"))
    
    url = "https://archive.org/stream/the-complete-works-of-h.-p.-lovecraft_202107/The%20Complete%20Works%20Of%20H.P.%20Lovecraft_djvu.txt"
    response = httpx.get(url)
    response.raise_for_status()
    fullpage =  response.text
    fullpage = fullpage.split('The Haunter of the Dark 00.0.2... 0c. ccccecccecceccecceecceccecceeececceeeceeceecaeeceececaueceeeeeaeeseeeeeseeeaees 694 ')[1]
    fullpage = "\n\n".join([x.strip() for x in fullpage.split('Return to Table of Contents ')[:-1]])

    train, _ = fullpage.split("""
The Dunwich Horror 
(1928) 


""")
    
    test = []

    test_texts = {
        "The Dunwich Horror": "https://www.hplovecraft.com/writings/texts/fiction/dh.aspx",
        "The Whisperer in Darkness": "https://www.hplovecraft.com/writings/texts/fiction/wid.aspx",
        "At the Mountains of Madness": "https://www.hplovecraft.com/writings/texts/fiction/mm.aspx",
        "The Shadow over Innsmouth": "https://www.hplovecraft.com/writings/texts/fiction/soi.aspx",
    }
    for title, url in test_texts.items():
        path = DATA / f"{title.lower().replace(' ', '_')}.txt"
        if path.exists():
            print(f"[data] {title} already exists, loading from disk.")
            text = path.read_text(encoding="utf-8")
            test.append((title, text))
            continue
        response = httpx.get(url)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        test.append((title, text))
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    return train, test

def get_orwell_text():
    train = ""
    GUTENBERG_AU_BASE_URL = "https://gutenberg.net.au/"
    books = {
        "Down and Out in Paris and London": "/ebooks01/0100171.txt",
        "Burmese Days": "/ebooks02/0200051.txt",
        "Coming up for Air": "/ebooks02/0200031.txt",
        "Animal Farm": "/ebooks01/0100011.txt",
    }
    DATA.mkdir(parents=True, exist_ok=True)
    for title, url in books.items():
        path = DATA / f"{title.lower().replace(' ', '_')}.txt"
        if path.exists():
            print(f"[data] {title} already exists, skipping download.")
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                train += text + "\n\n"
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

        train += text + "\n\n"

    test = []
    test_texts = {
        "A Hanging": "https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/a-hanging/",
        "Good Bad Books": "https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/good-bad-books/",
        "Politics and the English Language": "https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/politics-and-the-english-language/",
        "Shooting an Elephant": "https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/shooting-an-elephant/",
        "Why I Write": "https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/why-i-write/"
    }
    for title, url in test_texts.items():
        path = DATA / f"{title.lower().replace(' ', '_')}.txt"
        if path.exists():
            print(f"[data] {title} already exists, loading from disk.")
            text = path.read_text(encoding="utf-8")
            test.append((title, text))
            continue
        response = httpx.get(url)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        #Get the text inside the div with class "entry-content", except the first element
        content_div = soup.find("section", class_="entry-content")
        paragraphs = content_div.find_all("p")[1:]
        text = "\n\n".join([p.get_text() for p in paragraphs])
        test.append((title, text))
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return train, test

def get_chesterton_text():
    books = {
        "The Man Who Was Thursday": "https://www.gutenberg.org/cache/epub/1695/pg1695.txt",
        "The Innocence of Father Brown": "https://www.gutenberg.org/cache/epub/204/pg204.txt",
        "What's Wrong with the World": "https://www.gutenberg.org/cache/epub/1717/pg1717.txt",
        "Orthodoxy": "https://www.gutenberg.org/cache/epub/16769/pg16769.txt",
    }
    train = ""
    DATA.mkdir(parents=True, exist_ok=True)
    for title, url in books.items():
        path = DATA / f"{title.lower().replace(' ', '_')}.txt"
        if path.exists():
            print(f"[data] {title} already exists, skipping download.")
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                train += text + "\n\n"
            continue
        print(f"[data] Downloading {title} from {url}")
        response = httpx.get(url)
        response.raise_for_status()
        text = response.text

        # Remove the Gutenberg header and footer
        if "The Project Gutenberg EBook of" in text:
            text = text.split("The Project Gutenberg EBook of")[1]
            text = text.split("End of the Project Gutenberg EBook")[0]
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        train += text + "\n\n"

    test = []
    test_texts = {
        "January One": "https://www.chesterton.org/january-one-essay/",
        "Negative and Positive Morality": "https://www.chesterton.org/negative-and-positive-morality/",
        "On Mending and Ending Things": "https://www.chesterton.org/on-mending-and-ending-things/",
        "A Defence of Rash Vows": "https://www.chesterton.org/a-defence-of-rash-vows/",
    }

    for title, url in test_texts.items():
        path = DATA / f"{title.lower().replace(' ', '_')}.txt"
        if path.exists():
            print(f"[data] {title} already exists, loading from disk.")
            text = path.read_text(encoding="utf-8")
            test.append((title, text))
            continue
        response = httpx.get(url)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n\n".join([p.get_text() for p in paragraphs])
        test.append((title, text))
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return train, test

def build_vocab(*text):
    cv = CountVectorizer(stop_words='english', max_features=5000)
    counts = cv.fit_transform(text)
    df_counts = pd.DataFrame(counts.toarray(), columns=cv.get_feature_names_out())
    print(df_counts.head())
    return df_counts

def guess_author(vocab, *texts):
    cv = CountVectorizer(vocabulary=vocab.columns.tolist())
    counts = cv.fit_transform(texts)
    df_counts = pd.DataFrame(counts.toarray(), columns=cv.get_feature_names_out())
    #print(df_counts.head())
    similarities = cosine_similarity(vocab, df_counts)
    #print(similarities)
    return similarities

if __name__ == "__main__":
    lovecraft_train, lovecraft_test = get_lovecraft_text()
    orwell_train, orwell_test = get_orwell_text()
    chesterton_train, chesterton_test = get_chesterton_text()
    vocab = build_vocab(lovecraft_train, orwell_train, chesterton_train)
    print(f"Vocab size: {len(vocab)}")
    print(vocab)

    print("LOVECRAFT TEST")
    print("(similarities: [lovecraft, orwell, chesterton])")
    for title, text in lovecraft_test:
        sims = guess_author(vocab, text)
        print(f"{title}: {sims}")

    print("\nORWELL TEST")
    print("(similarities: [lovecraft, orwell, chesterton])")
    for title, text in orwell_test:
        sims = guess_author(vocab, text)
        print(f"{title}: {sims}")

    print("\nCHESTERTON TEST")
    print("(similarities: [lovecraft, orwell, chesterton])")
    for title, text in chesterton_test:
        sims = guess_author(vocab, text)
        print(f"{title}: {sims}")