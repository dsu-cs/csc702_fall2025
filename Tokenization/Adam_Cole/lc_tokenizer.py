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

# Something slightly longer, from one of H.P.'s penpals.
enc = tokenizer.encode("Hither came Conan, the Cimmerian, black-haired, sullen-eyed, sword in hand, a thief, a reaver, a slayer, with gigantic melancholies and gigantic mirth, to tread the jeweled thrones of the Earth under his sandalled feet")
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")

# Longer. Orwell. (_Down and Out in Paris and London_)
enc = tokenizer.encode("""
It is not a figure of 
speech, it is a mere statement of fact to say that a French 
cook will spit in the soup— that is, if he is not going to 
drink it himself. He is an artist, but his art is not cleanliness. 
To a certain extent he is even dirty because he is an 
artist, for food, to look smart, needs dirty treatment. When 
a steak, for instance, is brought up for the head cook’s in¬ 
spection, he does not handle it with a fork. He picks it up in 
his fingers and slaps it down, runs his thumb round the dish 
and licks it to taste the gravy, runs it round and licks again, 
then steps back and contemplates the piece of meat like an 
artist judging a picture, then presses it lovingly into place 
with his fat, pink fingers, every one of which he has licked a 
hundred times that morning. When he is satisfied, he takes 
a cloth and wipes his fingerprints from the dish, and hands 
it to the waiter. And the waiter, of course, dips HIS fingers 
into the gravy—his nasty, greasy fingers which he is for ever 
running through his brilliantined hair. Whenever one pays 
more than, say, ten francs for a dish of meat in Paris, one 
may be certain that it has been fingered in this manner. In 
very cheap restaurants it is different; there, the same trouble 
is not taken over the food, and it is just forked out of the pan 
and flung on to a plate, without handling. Roughly speaking, 
the more one pays for food, the more sweat and spittle one is obliged to eat with it.""".replace("\n", " "))
print(enc.n_sequences, enc.ids, enc.tokens, end="\n\n")