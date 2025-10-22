Intended to be a simple encoder only model that predicts missing words in a sentence.

I tumbled around trying to figure out what to do for a while and saw all sorts of possible uses for transformers. I ended up going for a bert flavored missing word prediction so I wouldn't need to worry about more complicated datasets and could malevolently use Dracula.txt again. 

I cobbled together some code from several tutorials and examples to make something that at least looks like it trains correctly. However, as it is now, it still spews out garbage when you try to actually predict something. Why it does this, I don't know yet. Perhaps I messed up in the prediction function, the transformer itself is all out of wack, or Dracula.txt still sucks as a dataset.