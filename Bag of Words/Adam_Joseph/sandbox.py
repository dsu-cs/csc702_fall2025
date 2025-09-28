#https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html

# CBOW Tensorflow
#https://spotintelligence.com/2023/07/27/continuous-bag-of-words/#Pre-trained_Continuous_Bag-of-Words_CBOW_embeddings

# Datasets

#Amazon reviews https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
#Social media posts https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))


df = pd.read_csv('data/sentiment_analysis.csv')

#https://stackoverflow.com/questions/59435472/implementing-bag-of-words-in-scikit-learn
cv = CountVectorizer(ngram_range=(2,2), max_features=1000)
tf = TfidfVectorizer(ngram_range=(1,1), max_features=1000)
counts = cv.fit_transform(df['text'])
tfidf = tf.fit_transform(df['text'])

df_counts = pd.DataFrame(counts.toarray(), columns=cv.get_feature_names_out())
df_tfidf = pd.DataFrame(tfidf.toarray(), columns=tf.get_feature_names_out())
#df_counts['text'] = df['text']
df_counts['label'] = df['sentiment']
df_tfidf['label'] = df['sentiment']

# Split both dataframes into train and test sets.
from sklearn.model_selection import train_test_split
X_train_counts, X_test_counts, y_train_counts, y_test_counts = train_test_split(df_counts.drop(columns=['label']), df_counts['label'], test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(df_tfidf.drop(columns=['label']), df_tfidf['label'], test_size=0.2, random_state=42)

print(X_train_counts.shape, X_test_counts.shape)
#print(df_counts.head())
print(df_tfidf.head())
#for sw in stop_words:
#    if sw in df_counts.columns:
#        df_counts.drop(columns=[sw], inplace=True)

print(df_counts.head())

#Get totals by label.
groups = df_tfidf.groupby('label').sum()
print(groups)

# Get the difference of "positive" - "negative"
diff = groups.loc['positive'] - groups.loc['negative']
diff.sort_values(ascending=False, inplace=True)

# Top "good" and "bad" words.
print(diff.head(20))
print(diff.tail(20))

#https://www.kaggle.com/datasets/tarkkaanko/amazon

df_amz = pd.read_csv('data/amazon_reviews/amazon_reviews.csv')
print(df_amz.head())

#Drop everything from df_amz except for overall and reviewText.
df_amz = df_amz[['overall', 'reviewText']]

#Drop rows with NaN values.
df_amz.dropna(inplace=True)
print(df_amz.head())
#label reviews with overall >= 4 as positive, <= 2 as negative, and 3 as neutral.
def label_review(overall):
    if overall >= 4:
        return 'positive'
    elif overall <= 2:
        return 'negative'
    else:
        return 'neutral'
df_amz['label'] = df_amz['overall'].apply(label_review)
#Drop overall column.
df_amz.drop(columns=['overall'], inplace=True)
#Drop neutral reviews.
df_amz = df_amz[df_amz['label'] != 'neutral']
print(df_amz['label'].value_counts())