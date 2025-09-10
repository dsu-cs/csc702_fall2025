Embeddings to Words

By Mike Kessler and Joseph Doty

We had several iterations of the code before this final one.
First we loaded the file and did some calculations on the numbers (Totals, Average, etc.)
That didn't provide anything interesting.

Then we decided to try to load the file as vectors and calculate word similarity.
This started to give good results.  We tested with several words and stuck with 5 that gave good results.

Then we tried to make a sentence, or at least string a few words together.
Taking the top 1 match was a problem as that ended up in a circular pattern.
We changed it to match the 2nd best word and then the results looked better.

These are the results when we ran the code:

 Similar words to: **economy**
  slowing         → similarity: 0.9236
  economic        → similarity: 0.8592
  downward        → similarity: 0.8438
  trend           → similarity: 0.8389
  sentiment       → similarity: 0.8038

 Similar words to: **baseball**
  sports          → similarity: 0.7798
  games           → similarity: 0.7610
  cbs             → similarity: 0.7369
  game            → similarity: 0.7226
  bartlett        → similarity: 0.7147

 Similar words to: **music**
  works           → similarity: 0.7645
  fit             → similarity: 0.7618
  spirit          → similarity: 0.7508
  playing         → similarity: 0.7461
  song            → similarity: 0.7409

 Similar words to: **computer**
  computers       → similarity: 0.8930
  software        → similarity: 0.8791
  digital         → similarity: 0.8574
  systems         → similarity: 0.8494
  machines        → similarity: 0.8390

 Similar words to: **is**
  problem         → similarity: 0.8085
  clearly         → similarity: 0.7709
  considering     → similarity: 0.7438
  politician      → similarity: 0.7303
  reality         → similarity: 0.7140

 5 Word Sentence starting after "economy"
economic monetary policy reform free


As an additional experiment, we tried to find a reduced dimenstion matrix that maintained the relationships between the words. 
Thinking about the approach now, it was a rather naive strategy. By creating a different embedding, perhaps a different model is needed to make it useful.

Here is an example from the experiment, the word was 'korea': 

Top 5 similar words original vector:
ministry: 0.7984
exports: 0.7675
abroad: 0.7631
soviet: 0.7618
south: 0.7445
Top 5 similar words reduced vectors:
region: 0.9045
key: 0.9021
export: 0.8898
the: 0.8848
nations: 0.8827

As we can see, there is some notion of a relationship between the embeddings with both 'export' and 'exports' appearing in the list. 
Other words do seem related like 'abroad' and 'region'. It might be possible that with some hyperparameter tuning a better solution can be found. 

