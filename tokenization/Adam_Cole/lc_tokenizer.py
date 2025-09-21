from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from pathlib import Path



if not Path("saved_tokenizers/lovecraft-vocab.json").exists():
    tokenizer = Tokenizer(BPE())
    #trainer = BpeTrainer()
    with open("../../words_to_emb/Ashar_Adam/data/lovecraft.txt", "r", encoding="utf-8") as f:
        tokenizer.train_from_iterator(f)

    # Or from a file:
    #tokenizer.train(["../../words_to_emb/Ashar_Adam/data/lovecraft.txt"])
    Path("saved_tokenizers").mkdir(parents=True, exist_ok=True)

    # Saves its work in lovecraft-merges.txt, which is fun.
    tokenizer.save("saved_tokenizers/lovecraft-vocab.json")
else:
    ...
    tokenizer = Tokenizer.from_file("saved_tokenizers/lovecraft-vocab.json")

# Not common enough to have its own token, but "goth" is.
enc = tokenizer.encode("Shoggoth")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")

# Common enough to have its own token.
enc = tokenizer.encode("Cthulhu")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")


enc = tokenizer.encode("It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")

# Something longer, from one of H.P.'s penpals.
enc = tokenizer.encode("Hither came Conan, the Cimmerian, black-haired, sullen-eyed, sword in hand, a thief, a reaver, a slayer, with gigantic melancholies and gigantic mirth, to tread the jeweled thrones of the Earth under his sandalled feet")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")