Uses boring old word2vec model to generate word embeddings. 

Cool part is in the merge function. We take two models trained on separate texts and attempt to add the out of vocab words to one another. An estimated vector for the new word is calculated from an average of similar words found in both vocabs.

Test texts are Dracula and Oliver Twist.

We look at similar words for both the original and updated models and found a slew of new similar words for something like "count" in Oliver Twist. Whether they make sense is up to the beholder. Note: "count" was already in the Oliver vocab, most of the new similar words are added from Dracula. Perhaps the meaning even got skewed a bit depending on how differently the word "count" is used in each book.
