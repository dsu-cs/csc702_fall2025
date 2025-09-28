# Word Similarities

Code generated with ChatGPT

This projects goal is to look into the changes in meanings of words overtime, and in different contexts. I used the full set of words from Moby Dick, Romeo and Juliet, and a more modern work A Modern Instance, which are about 250 years apart. They are also written in a very different style comparatively.

## Implementation

I loaded the 3 datasets in seperately, and trained each one to its own model using Word2Vec. I had to align the words in each dataset to get proper comparisons of the same word, this was accomplished with the orthogonal Procrustes method. I built similarity matrices to compare all the words to eachother for each corpus, as well as a delta matrix to see which words had the stronger relationship between the 3 corpuses.

## Romeo & Juliet

shows a clear poetic and emotional style, with words like "love", "death", and "heaven" are strongly associated with eachother, reflecting the romantic and tragic themes.
"cunning" and "death" are negatively correlated, indicating a contrast between human scheming and mortality.

## Moby Dick

"cunning", "death", and "heaven" are closely similar, suggesting connection with morality and grounded themes. "love" is weakely connected with most words, except for "soul, which highlights a more spiritual or moral framing.

## Modern Instance

shares a more grounded representation of love and death. Love and Soul are closely intertwined, as well as soul and cunning, hinting at the conflict between inner life and social scheming. Compared to Moby-Dick, A Modern Instance is more focused on personal relationships and destiny, because love-fate have a pretty close connection.

## Results Analysis

These results show the differences in the themes of each piece, with Romeo and Juliet being centered around love, emotion, and tragedgy, while Moby Dick is focused more on morality, human scheming, and nautical themes. A Modern Instance is more focused on psychological and social realism. This isn't a completely accurate way to look at the usage of words overtime because of the difference in themes, but it is interesting to see the difference in themes revealed by the word embeddings. It is also interesting to see a semantic shift of the meaning of the word love.
16th century: Love = passion + tragedy.
19th century (mid): Love and Death framed in cosmic/theological struggles.
19th century (late): Love and Fate framed in psychological and social contexts.

 Romeo & Juliet similarity matrix:
          love  death  fate    sea whale  heaven   lord   soul romance  cunning
love     1.000  0.197  None  0.191  None   0.179 -0.031 -0.049    None    0.032
death    0.197  1.000  None  0.049  None   0.047 -0.042  0.003    None   -0.308
fate       NaN    NaN  None    NaN  None     NaN    NaN    NaN    None      NaN
sea      0.191  0.049  None  1.000  None   0.174 -0.356  0.050    None    0.013
whale      NaN    NaN  None    NaN  None     NaN    NaN    NaN    None      NaN
heaven   0.179  0.047  None  0.174  None   1.000 -0.028 -0.091    None   -0.141
lord    -0.031 -0.042  None -0.356  None  -0.028  1.000  0.269    None    0.012
soul    -0.049  0.003  None  0.050  None  -0.091  0.269  1.000    None   -0.033
romance    NaN    NaN  None    NaN  None     NaN    NaN    NaN    None      NaN
cunning  0.032 -0.308  None  0.013  None  -0.141  0.012 -0.033    None    1.000

🔹 Moby-Dick similarity matrix:
          love  death   fate    sea  whale  heaven   lord   soul romance  cunning
love     1.000  0.042  0.050 -0.091  0.009   0.031  0.028  0.162    None    0.015
death    0.042  1.000  0.003  0.123 -0.036   0.190  0.104  0.222    None    0.185
fate     0.050  0.003  1.000 -0.080 -0.108   0.050  0.130 -0.043    None    0.158
sea     -0.091  0.123 -0.080  1.000  0.255   0.067  0.054  0.135    None    0.108
whale    0.009 -0.036 -0.108  0.255  1.000   0.024  0.151  0.103    None    0.037
heaven   0.031  0.190  0.050  0.067  0.024   1.000  0.238  0.179    None    0.320
lord     0.028  0.104  0.130  0.054  0.151   0.238  1.000  0.088    None    0.132
soul     0.162  0.222 -0.043  0.135  0.103   0.179  0.088  1.000    None    0.154
romance    NaN    NaN    NaN    NaN    NaN     NaN    NaN    NaN    None      NaN
cunning  0.015  0.185  0.158  0.108  0.037   0.320  0.132  0.154    None    1.000

🔹 A Modern Instance similarity matrix:
          love  death   fate    sea whale  heaven   lord   soul romance  cunning
love     1.000  0.203  0.211 -0.100  None   0.150  0.014  0.327    None    0.089
death    0.203  1.000  0.082 -0.087  None  -0.088  0.095  0.024    None    0.103
fate     0.211  0.082  1.000  0.113  None   0.072 -0.046  0.321    None    0.205
sea     -0.100 -0.087  0.113  1.000  None   0.241 -0.047  0.118    None    0.041
whale      NaN    NaN    NaN    NaN  None     NaN    NaN    NaN    None      NaN
heaven   0.150 -0.088  0.072  0.241  None   1.000 -0.007  0.201    None    0.035
lord     0.014  0.095 -0.046 -0.047  None  -0.007  1.000 -0.123    None    0.012
soul     0.327  0.024  0.321  0.118  None   0.201 -0.123  1.000    None    0.320
romance    NaN    NaN    NaN    NaN  None     NaN    NaN    NaN    None      NaN
cunning  0.089  0.103  0.205  0.041  None   0.035  0.012  0.320    None    1.000

🔹 Top 10 semantic shifts (Romeo vs Moby):
   death – cunning  | Δ = -0.494 (Moby stronger)
  heaven – cunning  | Δ = -0.461 (Moby stronger)
     sea – lord     | Δ = -0.411 (Moby stronger)
    love – sea      | Δ = 0.283 (Romeo stronger)
  heaven – soul     | Δ = -0.270 (Moby stronger)
  heaven – lord     | Δ = -0.266 (Moby stronger)
   death – soul     | Δ = -0.219 (Moby stronger)
    love – soul     | Δ = -0.211 (Moby stronger)
    soul – cunning  | Δ = -0.187 (Moby stronger)
    lord – soul     | Δ = 0.181 (Romeo stronger)

🔹 Top 10 semantic shifts (Romeo vs Modern):
   death – cunning  | Δ = -0.411 (Modern stronger)
    lord – soul     | Δ = 0.392 (Romeo stronger)
    love – soul     | Δ = -0.377 (Modern stronger)
    soul – cunning  | Δ = -0.353 (Modern stronger)
     sea – lord     | Δ = -0.309 (Modern stronger)
  heaven – soul     | Δ = -0.292 (Modern stronger)
    love – sea      | Δ = 0.292 (Romeo stronger)
  heaven – cunning  | Δ = -0.176 (Modern stronger)
   death – lord     | Δ = -0.136 (Modern stronger)
   death – sea      | Δ = 0.136 (Romeo stronger)

🔹 Top 10 semantic shifts (Moby vs Modern):
    fate – soul     | Δ = -0.364 (Modern stronger)
  heaven – cunning  | Δ = 0.285 (Moby stronger)
   death – heaven   | Δ = 0.278 (Moby stronger)
  heaven – lord     | Δ = 0.245 (Moby stronger)
    lord – soul     | Δ = 0.211 (Moby stronger)
   death – sea      | Δ = 0.209 (Moby stronger)
   death – soul     | Δ = 0.198 (Moby stronger)
    fate – sea      | Δ = -0.193 (Modern stronger)
    fate – lord     | Δ = 0.176 (Moby stronger)
     sea – heaven   | Δ = -0.174 (Modern stronger)
