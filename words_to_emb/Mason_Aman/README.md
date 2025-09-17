Overall Goal:

The goal with this project is to take English and German embedding vector sets, then maybe a dictionary and directly relate/translate English words to German words. So a word like ‘dog’ and the German ‘Hund’ may have a high similarity. These are basically the same thing. Other words that translate poorly may have 0.5 or lower. Then, relate the similarities and return a top list of best translations.

So for instance, say you have a word that doesn't translate that great, we are going to find what words in the target language best represent what we could use instead, if any.

Further, if time allows we will explore looking at surrounding words and see if word order/words used will dramatically change the similarity between languages. For instance, "Do you speak German?" Turns into "Sprechen Sie Deutsch?". This word-for-word translates to "Speak you german?". So we have some information change. But the meaning remains the same. So can we determine how similar sentences like these are, or possibly measure how similar sentences are? Can we measure if some sentences translate better than others?


The process we implemented for individual word comparison between languages:


The bulk of what was done for the single word translation was inside a class called SingleWordTranslator. This was initialized with sentence_transformers SentenceTransformer. This was something online that has much more potential than what I used it for, but I just used it to use a model 'sentence-transformers/distiluse-base-multilingual-cased-v1'. This models description is "It maps sentences & paragraphs to a 512 dimensional dense vector space and can be used for tasks like clustering or semantic search." -https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1. It is basically just for mapping (in my case words) to a vector space for processesing usage. A.K.A. it is a model that knows dog and Hund are the same thing. It normally is for sentences, but words work too. It trains on the same books in several languages and learns the sentence similarities (or word similarities).

Moving on, this class will basically create two embeddings (english and german) and then compares the cosine similarities between the source (input), and target embeddings (transpose of the other vocabularies vocab). Moreover, then, we basically rank the similarity scores and report the k (hardcoded to 5 here) most similar scores (and more importantly the accompanying words).

Later, some examples are used. One with a small (31 words each ish) vocabulary, and one with a large vocab (50k words each language). This gets into us running our translator. We basically just run it and give it the two embedding sets in the correct order. Finally, it prints the results.

I also implemented a translate_word function to try with the different langauges. This leads me into my final discussion point: results.

For the results, most of it was expected and good. For instance, with the small dataset, everything went exactly how expected and returned strongly correlated translations for everything (but I put exact translations in the vocabs so that is extremely expected). Furthermore, they had strong similarities (mid to high 0.9s). Additionally, with the larger datasets, interesting results were found. For instance, when the word "Schadenfruede" was translated, there were 5 words between 0.7725 and 0.8026 returned. This showed that there was no strong direct translation. This was expected because there really is not a great translation. Further, the word options were somewhat different as well showing how difficult translation can be occassionally. In contrast, we also tested an example that showed strong correlation: Ozean and it returned ocean, with a high (0.93) similarity score. This was what I was expecting to see.

There was one problem. When comparing English words to German words, sometimes English words would be included in the search results (in the german vocab) for the large datasets. My guess is that my function may be flawed, or the dataset may include English words because it is simply frequency counts from various articles. These articles (though in German), sometimes include english words because of how popular/common English is worldwide. That is my guess. It worked fine in the small dataset example though, which was good!

In sum, it seemed that there was a mysterious result with the English query in the larger dataset, but all other prompts had expected results that agreed with my hypotheses.