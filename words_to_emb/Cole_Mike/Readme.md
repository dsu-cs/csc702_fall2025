# 🧠 Words to Vectors
Modern LLM's are very good at translating text and word embeddings are a big part of that.  We decided to see if an old translated book would show if similar embeddings could be used to handle that task.
In this project we are comparing the embeddings for the English and the Spanish versions of Les Miserables.  We compare the top used words in each version and also compare plottings of the tokens.
We will look for patterns in word usage and related words.  In theory, a translated book would end up with similar embeddings.

We modified the files to only contain the text of the book.  
---

## 📦 Features

- Loads and tokenizes two text files
- Trains independent Word2Vec models using Skip-Gram
- Extracts top 10 most frequent words from each file
- Finds top 5 semantically similar words for each top word (within the same file)
- Visualizes top 100 word embeddings from each model using t-SNE

---

## 🛠 Requirements

- Python 3.7+
- `gensim`
- `scikit-learn`
- `matplotlib`

Install dependencies with:

pip install gensim scikit-learn matplotlib

---

## 📊 Output

Console: Lists top 10 frequent words in each file and their top 5 closest matches.

Plots: Two t-SNE visualizations showing top 100 word embeddings from each corpus.

---

## Analysis

The ten most common words compared along with the top 5 related words in each file will be compared to see how similar they are.

The Most common word was:

English                            Spanish (English translation)
🔹 'the' → Top matches:            🔹 'de' → Top matches:  (of)
   battle: 0.723                      hasta: 0.843  (until)
   second: 0.709                      en: 0.832  (in)
   field: 0.708                       calle: 0.811  (street)
   court: 0.702                       misma: 0.806  (same)
   station: 0.699                     ángulo: 0.802  (angle)

The most common word is different between the two books.    We found it odd that 'of' would be the most common word in a book.

The 2nd most common word is
🔹 'of' → Top matches:            🔹 'la' → Top matches:  (the)
   whole: 0.707                      una: 0.836  (a)
   grand: 0.647                      esta: 0.801  (this)
   spirit: 0.642                     aquella: 0.773  (that)
   public: 0.637                     su: 0.757  (his)
   ages: 0.633                       toda: 0.749  (all)

The top two common words were opposite in each file.  They could have very close counts.  The interesting thing is that none of the closely related words are close to being the same.
The Spanish version has much more common words related where the English has more unique words related.

The 3rd most common word is
   
🔹 'and' → Top matches:          🔹 'el' → Top matches:  (he)
   sobbing: 0.755                  este: 0.884  (this)
   hastily: 0.754                  del: 0.880  (of the)
   ensued: 0.754                   un: 0.834  (a)
   pause: 0.753                    al: 0.834  (to the)
   closely: 0.752                  otro: 0.763  (other)

4th

🔹 'to' → Top matches:          🔹 'que' → Top matches:  (that)
   away: 0.723                      cuanto: 0.841  (how much)
   order: 0.674                     aunque: 0.833  (although)
   thither: 0.672                   posible: 0.823  (possible)
   should: 0.658                    cual: 0.821  (which)
   denounce: 0.652                  fuerte: 0.817  (strong)

5th

🔹 'in' → Top matches:           🔹 'en' → Top matches: (in)
   new: 0.600                        hasta: 0.863  (until)
   whole: 0.585                      de: 0.832  (of)
   public: 0.582                     llegó: 0.817  (arrive)
   under: 0.579                      desde: 0.817  (from)
   thoughts: 0.571                   misma: 0.813  (same)

The 5th most common word is the same but none of the closely related words are.

6th

🔹 'he' → Top matches:            🔹 'se' → Top matches:  (HE)
   leblanc: 0.814                    le: 0.840  (you)
   marius: 0.812                     mientras: 0.833 (while)
   montparnasse: 0.793               sentía: 0.830  (felt)
   javert: 0.787                     hablaba: 0.827  (spoke)
   gavroche: 0.778                   vuelto: 0.827  (turned)
   
7th

🔹 'was' → Top matches:          🔹 'un' → Top matches:  (a)
   existed: 0.684                    este: 0.887  (this)
   became: 0.682                     aquel: 0.859  (that)
   constantly: 0.676                 el: 0.834  (he)
   deeply: 0.674                     gran: 0.791  (great)
   recalled: 0.674                   del: 0.763  (of the)

8th

🔹 'that' → Top matches:        🔹 'los' → Top matches:  (the)
   absolutely: 0.781                sus: 0.821  (their)
   perhaps: 0.773                   esos: 0.787  (those)
   certainly: 0.770                 todos: 0.785  (all)
   actually: 0.756                  aquellos: 0.781  (those)
   dying: 0.754                     estos: 0.759  (these)

9th

🔹 'it' → Top matches:         🔹 'no' → Top matches: (No)
   ill: 0.768                      nada: 0.850  (nothing)
   absolutely: 0.767               mucho: 0.839  (a lot)
   mistake: 0.765                  nadie: 0.823  (nobody)
   hurry: 0.758                    porque: 0.822  (because)
   easy: 0.757                     leer: 0.821  (read)

10th

🔹 'his' → Top matches:        🔹 'su' → Top matches:  (his)
   mabeuf: 0.663                  aquella: 0.801  (that)
   whose: 0.652                   cierta: 0.781  (true)
   seized: 0.644                  hacia: 0.759  (toward)
   fixed: 0.642                   la: 0.757  (the)
   pocket: 0.641                  alta: 0.756  (high)

A few interesting things emerged.  Only the 5th, 6th and 10th most common words were in the same positions in both books.  There is no similarity between the common words and their related words.
Another interesting find is that the Spanish related words are much more closely related that the English words.  That could indicate that the Spanish version may have a smaller set of words to relate or it could be related to the Spanish language use.

The Plottings show a similar finding.   The English words are spread out and not very closely related.   The Spanish version shows all the words are much more closely related as they are all grouped together in the plot.




   
   
   
   
   

   
   
   
   
   
   
   
   
   
   
