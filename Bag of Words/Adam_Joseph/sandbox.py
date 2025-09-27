#https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html

# CBOW Tensorflow
#https://spotintelligence.com/2023/07/27/continuous-bag-of-words/#Pre-trained_Continuous_Bag-of-Words_CBOW_embeddings

# Datasets

#Amazon reviews https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
#Social media posts https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import fasttext

df = pd.read_csv('data/sentiment_analysis.csv')

#https://stackoverflow.com/questions/59435472/implementing-bag-of-words-in-scikit-learn
cv = CountVectorizer()
counts = cv.fit_transform(df['text'])

df_counts = pd.DataFrame(counts.toarray(), columns=cv.get_feature_names_out())
#df_counts['text'] = df['text']
df_counts['label'] = df['sentiment']

print(df_counts.head())

#Get totals by label.
groups = df_counts.groupby('label').sum()
print(groups)

# Get the difference of "positive" - "negative"
diff = groups.loc['positive'] - groups.loc['negative']
diff.sort_values(ascending=False, inplace=True)

# Top "good" and "bad" words.
print(diff.head(20))
print(diff.tail(20))

# https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
# Load train.ft.txt.bz2 into a pandas dataframe
#df_fasttext = pd.read_csv('data/train.ft.txt.bz2', sep=' ', header=None, quoting=3)
#df_fasttext.columns = ['label'] + ['word_' + str(i) for i in range(1, df_fasttext.shape[1])]
#df_fasttext['label'] = df_fasttext['label'].map({'__label__positive': 'positive', '__label__negative': 'negative'})

#print(df_fasttext.head())