For this project, we decided to examine the inner mechanisms of Transformers, and how adjusting the configuration of a transformer (e.g. how many transformer block layers inside, or number of attention heads, or embedding dimension size (vector size)).

For this, our initial process included setting up a transformer architecture using some outside libraries, and running an example 'baseline' transformer to compare various other configurations against. 

Initially, we read in our dataset text and use word tokenization (later byte pair encoding tokenization for subword tokenization). The latter subword tokenization is then encoded. 

Next, we setup our TransformerLM. This heavily utilizes PyTorch. We create embeddings, PEs, along with the rest of the generic transformer process to create our transformer. 

Lastly, we train our transformers on the dataset. This is then compared against the differrent configurations of the same transformer trained on the same data to determine optimal configuration directions. This word 'directions' must be emphasized because we didn't have the time to test magnification of increasing/decreasing various variables, but as we get into later in the testing process, we just look for what kinds of things generally improve our performance on this dataset (using our transformer).

So, for the actual comparisons, we look at single variable and all variable changes in both increasing and decreasing the number of embedding dimensions, number of attention heads, and number of layers. So we compare increasing or decreasing either one of these, or all of these. And these 8 results are compared to our baseline to determine how altering these various components of the transformer later affects performance. In other words, if we turn this hyperparameter/variable up/down and then train this transformer on data and test it it will perform worse/better than the baseline variation.

Also, we used 1 epoch, batch size of 512, lr of 1e-3 and seq_len of 64 for varing reasons. We tried to emphasize speed because of so many configurations to test on my poor CPU. But also the lr was somewhat random. These shouldn't matter though as long as they are the same just to determine effectiveness of the other variables we are testing.


Hypothesis:

As for my predictions for what this would likely do, before seeing the results, I am assuming that increasing any of the parameters will increase performance because it will be more complex and able to handle more information in a more in-depth way of understanding?

    Due to time constraints and initial results looking odd, I decided to swap to just multivariable changes. I also decreased batch size, increased epochs, and retrained the baseline.

------------------

Okay, I ended up swapping to a GPU part way through this because it was set to take like two hours, so now I started getting more full results.

------------------

Analysis:

This now gave three versions. A baseline, a simpler model with fewer layers, heads, and embeddings more simple. Then there was the more complex model. After retraining with more epochs and all the explained differences, differing and somewhat interesting (though not super extreme) results were reported. 

So, the baseline started out, and this will be used to compare to the more simplistic model first. As hypothesized, this ended up performing slightly (though relatively) notable results. As it started out, the results were nearly identical on the first epoch. The second epoch showed them also nearly identical. As the third epoch came, the more simplistic model performed about 5% worse. Then, on the fourth epoch it compared about 14% worse relatively. Finally, it performed about 5% worse on the fifth epoch. 

Conversely, the more complex model showed very similar results, even worse results (barerly) on the first two epochs. For the third, however, it performed about 5% better than the baseline and much better than the simple model. For the fourth epoch, it performed slightly better than the baseline and about 15% better than the simple model. Finally, the last epoch showed it about 5% better than the simple model and slightly (and barely) better than the baseline. 

These results generally showed after a couple epochs (after they had learned), that more complex models were slightly better. Although with that in mind, these differences were very slight, and likely on more complex data and larger datasets, this would be seen even more extremely. Furthermore, It shows after 5 epochs they are all similar although they show differences. Again, after a certain amount of training I am guessing the models can only get so accurate. 

This all probably can be scaled out with bigger, more complex transformers. However, I believe it shows the basic concept that generally increasing the number of layers, embedding dimensionality, and number of heads will probably/hopefully improve performance. Likewise, when decreasing these, performance will worsen. 

Of course, the real world can have all sorts of exceptions to this potentially. And there may be caps for optimal performance, I don't really know but there may be.

** Loss refers to average loss in this document. Final loss is also recorded in this, but I use average.