import gensim
from gensim.models import Word2Vec

import numpy as np 
from numpy.linalg import norm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import optuna

nltk.download('punkt')

nltk.download('punkt_tab')

nltk.download('stopwords')

#Process all words and tokenize them from donquixote english version

def processed_english(english_text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(english_text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

file_names_english = ["DonQuixote_English.txt", "Hamlet_English.txt", "TreasureIsland_English.txt"]

# Process all files
processedenglish = []

for file_name in file_names_english:
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()
        tokens = processed_english(text)
        processedenglish.append(tokens)



#Process all words and tokenize them from donquixote spanish version

def processed_spanish(spanish_text):
    stop_words = set(stopwords.words('spanish'))
    tokens = word_tokenize(spanish_text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


# Process all files

file_names_spanish = ["DonQuixote_Spanish.txt", "Hamlet_Spanish.txt", "TreasureIsland_Spanish.txt"]

processedspanish = []

for file_name in file_names_spanish:
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()
        tokens = processed_spanish(text)
        processedspanish.append(tokens)

word_pairs = [
    ('quixote', 'quijote'),
    ('adventure', 'aventura'),
    ('knight', 'caballero'),
    ('treasure', 'tesoro'),
    ('island', 'isla'),
    ('ship', 'barco'),
    ('ghost', 'fantasma'),
    ('sword', 'espada'),
    ('king', 'rey'),
    ('queen', 'reina'),
]
#Optuna training

#find difference between words in english and in spanish to see if pattern emerges

def cosine_sim(eng1, span1, eng2, span2):

    diff1 = eng1 - span1
    print(diff1)

    diff2 = eng2 - span2
    print(diff2)

    cosine = np.dot(diff1, diff2) / (norm(diff1) * norm(diff2))
    print(cosine)

def objective(trial):
    vector_size = trial.suggest_int('vector_size', 50, 300)
    window = trial.suggest_int('window', 2, 10)
    min_count = trial.suggest_int('min_count', 1, 5)
    sg = trial.suggest_int('sg', 0, 1)
    epochs = trial.suggest_int('epochs', 5, 50)


#train and save both word2vec models

    english_model = Word2Vec(sentences=processedenglish, vector_size=vector_size, window=window, min_count=min_count, workers=4, sg=sg, epochs=epochs)

    spanish_model = Word2Vec(sentences=processedspanish, vector_size=vector_size, window=window, min_count=min_count, workers=4, sg=sg, epochs=epochs)

    scores = []

#implement optuna for hyperparameter optimization?

    for (eng_word, span_word) in word_pairs:
        try:
            eng_vector = english_model.wv[eng_word]
            span_vector = spanish_model.wv[span_word]

            idx = word_pairs.index((eng_word, span_word))
            next_indx = (idx + 1) % len(word_pairs)
            eng_vector2 = english_model.wv[word_pairs[next_indx][0]]
            span_vector2 = spanish_model.wv[word_pairs[next_indx][1]]

            diff1 = eng_vector - span_vector
            diff2 = eng_vector2 - span_vector2

            cosine = np.dot(diff1, diff2) / (norm(diff1) * norm(diff2))
            scores.append(cosine)
        except KeyError:
            continue
    
    if not scores:
        return -1.0
    
    return np.mean(scores)
    

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials = 100)

best_params = study.best_params
print("Nest hyperparameters:", best_params)

# Train and save final models
english_model = Word2Vec(
    sentences=processedenglish,
    vector_size=best_params['vector_size'],
    window=best_params['window'],
    min_count=best_params['min_count'],
    workers=4,
    sg=best_params['sg'],
    epochs=best_params['epochs']
)
english_model.save("english.model")

spanish_model = Word2Vec(
    sentences=processedspanish,
    vector_size=best_params['vector_size'],
    window=best_params['window'],
    min_count=best_params['min_count'],
    workers=4,
    sg=best_params['sg'],
    epochs=best_params['epochs']
)
spanish_model.save("spanish.model")

# Optional: check final cosine similarity average
def evaluate_model(model_en, model_es, word_pairs):
    scores = []
    for (eng_word, span_word) in word_pairs:
        try:
            eng_vector = model_en.wv[eng_word]
            span_vector = model_es.wv[span_word]

            idx = word_pairs.index((eng_word, span_word))
            next_idx = (idx + 1) % len(word_pairs)

            eng_vector2 = model_en.wv[word_pairs[next_idx][0]]
            span_vector2 = model_es.wv[word_pairs[next_idx][1]]

            diff1 = eng_vector - span_vector
            diff2 = eng_vector2 - span_vector2

            cosine = np.dot(diff1, diff2) / (norm(diff1) * norm(diff2))
            scores.append(cosine)
        except KeyError:
            continue
    return np.mean(scores) if scores else -1.0


final_score = evaluate_model(english_model, spanish_model, word_pairs)
print("Final average cosine similarity across word pairs:", final_score)


