# Word Similarities

Code generated with ChatGPT

This projects goal is to look into the changes in meanings of words overtime, and in different contexts. I used the full set of words from Moby Dick, Romeo and Juliet, and a more modern work A Modern Instance, which are about 250 years apart. They are also written in a very different style comparatively.
    I loaded the 3 datasets in seperately, and trained each one to its own model using Word2Vec. I had to align the words in each dataset to get proper comparisons of the same word, this was accomplished with the orthogonal Procrustes method. I built similarity matrices to compare all the words to eachother for each corpus, as well as a delta matrix to see which words had the stronger relationship between the 3 corpuses.
Romeo & Juliet:
shows a clear poetic and emotional style, with words like "love", "death", and "heaven" are strongly associated with eachother, reflecting the romantic and tragic themes.
"cunning" and "death" are negatively correlated, indicating a contrast between human scheming and mortality.
Moby Dick:
"cunning", "death", and "heaven" are closely similar, suggesting connection with morality and grounded themes. "love" is weakely connected with most words, except for "soul, which highlights a more spiritual or moral framing.
Modern Instance  shares a more grounded representation of love and death. Love and Soul are closely intertwined, as well as soul and cunning, hinting at the conflict between inner life and social scheming. Compared to Moby-Dick, A Modern Instance is more focused on personal relationships and destiny, because love-fate have a pretty close connection.
These results show the differences in the themes of each piece, with Romeo and Juliet being centered around love, emotion, and tragedgy, while Moby Dick is focused more on morality, human scheming, and nautical themes. A Modern Instance is more focused on psychological and social realism. This isn't a completely accurate way to look at the usage of words overtime because of the difference in themes, but it is interesting to see the difference in themes revealed by the word embeddings. It is also interesting to see a semantic shift of the meaning of the word love.
16th century: Love = passion + tragedy.
19th century (mid): Love and Death framed in cosmic/theological struggles.
19th century (late): Love and Fate framed in psychological and social contexts.
