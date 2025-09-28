import re
import nltk
import unicodedata
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from pathlib import Path

# nltk.download('punkt')
# nltk.download('punkt_tab')

# apostrophe types to fix
APOSTROPHES = [
    "\u2019", "\u2018", # curly quotes ’ ‘
    "\u02BC", "\u02BB", # modifier apostrophes ʼ ʻ
    "\u2032", "\u201B", "\uFF07", # primes / fullwidth
    "`", "ʹ", "ʾ", "ʼ"
]
# regex for words to keep leading, internal, and trailing apostrophes
WORD_RE = re.compile(r"(?:'[A-Za-z]+|[A-Za-z]+(?:['-][A-Za-z]+)*'?)")


# turns all apostrophe variants into '
def normalize_apostrophes(text):
    text = unicodedata.normalize("NFKC", text)
    for variant in set(APOSTROPHES):
        text = text.replace(variant, "'")
    return text


# return list of tokens preserving apostrophes and hyphens
def dialect_tokenize_sentence(sentence):
    return WORD_RE.findall(sentence)


# takes in path of text file and returns the tokenized sentences
def preprocess_file(path):
    # pull the text out in an overkill way in order to preserve apostrophes
    with open(path, "rb") as f:
        raw_bytes = f.read()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    text = normalize_apostrophes(raw_text)
    sentences = sent_tokenize(text)
    tokenized_sentences = [ [t.lower() for t in dialect_tokenize_sentence(s)] for s in sentences if s.strip()]
    return tokenized_sentences


# takes in folder name and returns paths of all .txt files in it
def list_txt_files(folder):
    folder_path = Path(folder)
    return [str(p) for p in folder_path.glob("*.txt")]


# takes in a folder's name and returns the tokenized sentences for all .txt files within it
def preprocess_and_compile_folder(folder):
    paths = list_txt_files(folder)
    compiled_sentences = []
    for path in paths:
        # print(path) # ensure all files getting appended to compiled list
        compiled_sentences.extend(preprocess_file(path))
    return compiled_sentences


# loads words from list of 10000 common english words
def load_common_words():
    with open('google-10000-english.txt', encoding='utf-8') as f:
        text = f.read()
    words = text.split()
    return set(words)


# find closest common english word to word from vernacular utilzing cosine similarity in the model's embedding space
def find_closest_common_word(word, common_words, model):
    # ensure word is in model's vocab
    if word not in model.wv:
        raise ValueError(f"user input: {word} is not in the given model's vocabulary")
    # return word if it is common words list
    if word in common_words:
        return word

    # returns ranked list of closest embeddings based on cosine similarity
    word_neighbors = model.wv.most_similar(word, topn=1000)

    # find the closest replacement word in the common words list
    for canidate, score in word_neighbors:
        if canidate in common_words:
            return canidate

    raise RuntimeError("No suitable substitute found")


# saves model's vocab to file for visual inspection
def save_vocab_to_file(model, filename="southern_vocab.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for word in model.wv.index_to_key:
            f.write(word + "\n")


# return closest words in model's vocabulary
def print_n_closest_model_neighbors(word, n, model):
    # ensure word is in model's vocab
    if word not in model.wv:
        raise ValueError(f"user input: {word} is not in the given model's vocabulary")
    
    # cap n at 100
    if n > 100:
        n = 100

    neighbors = model.wv.most_similar(word, topn=n)
    print(f"{word}'s nearest {n} neighbors:")
    for neighbor in neighbors:
        print(f"\t\t{neighbor}")
    


if __name__ == "__main__":
    # load in common words to compare with
    common_words = load_common_words()

    '''
    # create word2vec model of words from books with southern dialect
    southern_model = Word2Vec (
        sentences=preprocess_and_compile_folder('southern_vernaculars'),
        vector_size=200,
        window=10,
        min_count=2,
        workers=4,
        sg=1, # 0->CBOW, 1->Skip-gram
        seed=42,
    )
    southern_model.save('models/southern_vernacular_word2vec.model')
    '''
    
    # load in saved southern file
    southern_model = Word2Vec.load('models/southern_vernacular_word2vec.model')

    # test southern words
    southern_test_words = ['dat', '\'em', 'doan', 'de', 'en', 'gwyne']
    print(' ================================================= ')
    print(' === finding similar words to southern dialect === ')
    print(' ================================================= ')
    for word in southern_test_words:
        print(f"closest common word to {word}: {find_closest_common_word(word,common_words,southern_model)}")
        print_n_closest_model_neighbors(word, 5, southern_model)


    '''
    # create word2vec model of words from george macdonald collection
    scots_model = Word2Vec (
        sentences=preprocess_and_compile_folder('george_macdonald'),
        vector_size=200,
        window=10,
        min_count=2,
        workers=4,
        sg=1, # 0->CBOW, 1->Skip-gram
        seed=42,
    )
    scots_model.save('models/scots_vernacular_word2vec.model')
    '''
    # load in scots model
    scots_model = Word2Vec.load('models/scots_vernacular_word2vec.model')

    # test scots words
    scots_test_words = ['hae', 'o\'', 'oor', 'ye', 'ane', 'sae']
    print(' ================================================= ')
    print(' ==== finding similar words to scots dialect  ==== ')
    print(' ================================================= ')
    for word in scots_test_words:
        print(f"closest common word to {word}: {find_closest_common_word(word,common_words,scots_model)}")
        print_n_closest_model_neighbors(word, 5, scots_model)

