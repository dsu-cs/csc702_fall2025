Overall Goal:

The goal with this project is to take English and German embedding vector sets, then maybe a dictionary and directly relate/translate English words to German words. So a word like ‘dog’ and the German ‘Hund’ may have a high similarity. These are basically the same thing. Other words that translate poorly may have 0.5 or lower. Then, relate the similarities and return a top list of best translations.

So for instance, say you have a word that doesn't translate that great, we are going to find what words in the target language best represent what we could use instead, if any.

Further, if time allows we will explore looking at surrounding words and see if word order/words used will dramatically change the similarity between languages. For instance, "Do you speak German?" Turns into "Sprechen Sie Deutsch?". This word-for-word translates to "Speak you german?". So we have some information change. But the meaning remains the same. So can we determine how similar sentences like these are, or possibly measure how similar sentences are? Can we measure if some sentences translate better than others?



The process we implemented for individual word comparison between languages:



