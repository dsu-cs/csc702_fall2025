# Word Similarities

Code generated with ChatGPT

This projects goal is to look into the changes in meanings of words overtime, and in different contexts. I used the full set of words from Moby Dick and Romeo and Juliet, which are about 250 years apart. They are also written in a very different style comparatively.
    I loaded both datasets in seperately, and trained each one to its own model using Word2Vec. I had to align the words in each dataset to get proper comparisons of the same word, this was accomplished with the orthogonal Procrustes method. I built similarity matrices to compare all the words to eachother for each corpus, as well as a delta matrix to see which words had the stronger relationship between the 2 corpuses.
Romeo & Juliet:
shows a clear poetic and emotional style, with words like "love", "death", and "heaven" are strongly associated with eachother, reflecting the romantic and tragic themes.
"cunning" and "death" are negatively correlated, indicating a contrast between human scheming and mortality.
Moby Dick:
"cunning", "death", and "heaven" are closely similar, suggesting connection with morality and grounded themes. "love" is weakely connected with most words, except for "soul, which highlights a more spiritual or moral framing.
These results show the differences in the themes of each piece, with Romeo and Juliet being centered around love, emotion, and tragedgy, while Moby Dick is focused more on morality, human scheming, and nautical themes. This isn't a completely accurate way to look at the usage of words overtime because of the difference in themes, but it is interesting to see the difference in themes revealed by the word embeddings.