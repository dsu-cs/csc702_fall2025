# Transformer-Based Chess Bot: Expanding Attention

## This project updates the work of Sam and Wyatt's work on transformers. All modifications are listed under the `UPDATES` heading

With this project, we attempted to make a chess bot using a transformer decoder to predict the next move in the game. Our `preprocessing.ipynb` handled pulling in our data from Lichess and converting it into `.pt` files that our model later used. The `model.ipynb` has the code necessary for hyperparameter tuning and training the model (we had a major issue with the vm that caused us to lose the model we had trained for ~14 hours less than 12 hours before the submission deadline, so our final model is likely not as useful as it could have been). The `testing.ipynb` file shows our model's accuracy and some other useful metrics, and the `play.py` file provides the ability to play a game against the bot.

### How we use our data
The data we pull from Lichess comes in `.pgn` format and essentially captures recorded games move-by-move. We break these moves down into tokens and give them embeddings. We took a simple/naive approach to this that can be much improved upon to provide more meaningful context about a move to the model; however, working with our time constraint, this choice made the most sense. Each game then can be treated similar to a sentence, where one string representing a game can produce a large number of training data to the model. 

### How our model works
The model is simply doing next word prediction, only it is trained on chess moves as its vocabulary rather than a spoken language. It essentially is a next word predictor and displays why it might be that ChatGPT was very good at cheating when you would try to play it in chess in years past. 

We did find that our model is very good at predicting the next move in the testing dataset, as can be seen in our `testing.ipynb`; however, there is a chance it is cheating and using training data in that. That said, being able to predict the next move does not necessarily translate to being good at playing against a person.

### How to play against it
To play against our bot, run `python3 play.py` to pull up the game. The script will ask what color you are playing with, then enter `w` or `b`. User moves are taken in algebraic chess notation and the script will output the board in the terminal along with the bot's move. 

Our bot is yet to be mated by Chess.com's Martin bot; however, that is in large part due to the fact that it frequently fails to produce a legal move and has to resign.

### Extra files
We have a number of other files that are not included in our submission to github, solely because of their size. These include our training data pulled from Lichess which is 46Gb and our encoded game tensors that were actually used for training.


# UPDATES:

Files worked on: README.md, model_with_extended_attention.ipynb, ...

For the model_with_extended_attention.ipynb file, what we are attempting to do is recreate the prior decoder-only architecture while maintaining the ability for hyperparameter tuning in a similar way. Then have a later analysis/comparison to the baseline model trained and tested in the prior version that Wyatt worked on. More specifically, this architecture will be adjusted, trained, and compared to the prior version (baseline). This variation will have various 'upgrades' to the baseline architecture, addressing concerns revolving around the attention mechanism.

We attempt to add two different features to the decoder architecture: Quantization and Sparse Attention. This will aim to improve efficiency of the attention mechanisms. Hopefully accuracy/metrics will also improve by adding these features as well, or at the very least decrease time taken.

We use the bitsandbytes library for Quantization. 

The new ipynb file contains notes and code, but especially notable is that you need a GPU for this. The code will then run through cells that do some imports, check versions of those imports, create our dataset, and then the good stuff!

It then moves on to create sparse multihead attention self-attention layers with a sliding window to make the attention mechanism more efficient.

Then, the SparseDecoderLayer throws these layers in with the rest of a decoder layer for the total sparse attention layer.

Finally, this is put into the overall decoder. This is then quantized in 8-bits. This replaces the linear layers with quantized layers.

Finally, an example cell trains with various quicker hyperparameters to test out. From here, we will now train the new architecture using the previous hyperparameters/setup as close as we can and compare the results!


### New Architecture Results:

* We find that training time takes a significant hit with a time increase of nearly 6x per epoch.
* Testing results in a useless model. We observe a test loss of 520+ and an accuracy of less than 1.5%
* The play file was abandoned because the bot guessing legal moves was not in the question. 


### New Architecture Analysis:
There is obviously an issue in our training pipeline. Our assumption is that there is an issue with how our previous loss function interacts with the quantization of the model. However, due to time constraints, particularly that posed by training, we are unable to dig deeper into this in time to meet our deadline, despite being proactive with the project. 

### NOTE:
The model file exceeds the storage capacity of GitHub's regular uploads, so it is not included in our push to the repository.


