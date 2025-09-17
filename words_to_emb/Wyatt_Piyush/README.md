# Project: Odd dialect translation
### Authors: Wyatt & Piyush
## Idea: 
Train a Word2Vec model to create embeddings based around texts with dialects that are difficult to read and find the nearest word embedding that represents a word in a list of common English words to translate into a more commonly understandable text.

## Process:
### Setup
One extra setup step apart from imports is the need to run `nltk.download('punkt')` and `nltk.download('punkt_tab')` in a python script in order to allow the tokenize library to function properly. 

We first had to alter how we tokenized words so that sentences such as  
> "Den I reck'n'd I' inves' de thirty-five dollars right off en keep things a movin'," 

wouldn't be broken down into letters alone. 

From there we compiled all books we had gathered for both southern and Scots dialects, the two dialects we planned to focus on. We gathered 4 novels that were notable for their thick southern accents and 13 novels by George MacDonald for our Scots model. 

We realized that just grabbing some of the nearest word embeddings might leave us with similarly confusing words, so we grabbed our `google-10000-english.txt` file from [this GitHub Page](https://github.com/first20hours/google-10000-english/blob/master/20k.txt) in order to have an easy way to check if the word was a commonly used English word. 

### How it (should) work
When the user passes a word to the `find_closest_common_word()` function for a given model, it will create a sorted list of the closest word embeddings via cosine similarity to the input as "candidates" to be the translated word. Then, it traverses through that list until it finds the closest word that is also in the `google-10000-english.txt` file to ensure that it is not a similarly tricky word to read. The function then returns this word as the translation.

### Our hope
We planned to then implement a way for sentences to be translated by copying portions of the original texts to translate them into something more straighforward and readable. It was to be a crude implementation, only translating one word at a time, but that could be enough to be a useful tool when reading.

## Results
We found that in practice trying to translate nonsensical words resulted in even more nonsensical words. You can review our `output.txt` file to see that the closest common word to "doan," which we hope is "don't," turns out to be "kin." Similarly the word "gwyne" produces "um" when we hope to see "going." Despite there being 3 times more text for Scots to train on, both models performed similarly poorly. 

Although we tried, we did not find that tuning the hyperparameters could increase our understanding of the models' outputs. The final models used vectors of length 200, a window size of 10, and skip-gram as opposed to CBOW.

### Why we think this failed
Obviously, more text to train our models on would have been better. What might be more problematic though is the dictionary that the models end up with. Words like "reck'n'd" only show up once in all of the text, and there are more than 5 different ways that "reckoned" is spelled throughout the dialouge of one book. That isn't the case for every word though; "gwyne" shows up multiple times across books. Regardless, there is too much inconsistency for these words to have enough attention to get the embeddings they need in order to match them up closely with the common words they represent. 

Another important reason is that these words are not found isolated within more coherent text. They are always found within dialouge, most of which is filled with other, simillarly chopped up words. So, they don't have solid words to reference around them. 

## Extra Note
We planned to make this an interactive program; however, due to its poor performance we instead ran some sample words and displayed the word our program chose, along with the top 5 closest words within the model for both the southern and Scots dialect. 