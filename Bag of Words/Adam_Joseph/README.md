## sandbox.py

Some experimentation with Kaggle datasets with an idea to do some kind of sentiment analysis. Decided to go in a different direction.

## guess.py

A toy that tries to guess the author of a work by studying differences in vocabulary used.

### Process

It first loads some representative texts by H.P. Lovecraft, George Orwell, and G.K. Chesterton and identifies the top 5000 words in that corpus. It then compares a number of other texts by those authors with the vocabulary distributions of the first texts.

### Issues

The data collection process needs more work. In particular, we're keeping a lot of publication years in the data.

### Results/Output:

```
LOVECRAFT TEST
(similarities: [lovecraft, orwell, chesterton])
The Dunwich Horror: [[0.77576191]
 [0.63931151]
 [0.58324663]]
The Whisperer in Darkness: [[0.74812134]
 [0.62562743]
 [0.61531909]]
At the Mountains of Madness: [[0.69973479]
 [0.49451228]
 [0.46836992]]
The Shadow over Innsmouth: [[0.75967638]
 [0.66997967]
 [0.5917539 ]]

ORWELL TEST
(similarities: [lovecraft, orwell, chesterton])
A Hanging: [[0.3750017 ]
 [0.48936932]
 [0.48474637]]
Good Bad Books: [[0.256608  ]
 [0.31056167]
 [0.29650626]]
Politics and the English Language: [[0.29101204]
 [0.3707228 ]
 [0.36554944]]
Shooting an Elephant: [[0.40162381]
 [0.48732276]
 [0.41446885]]
Why I Write: [[0.35467181]
 [0.43325073]
 [0.3930558 ]]

CHESTERTON TEST
(similarities: [lovecraft, orwell, chesterton])
January One: [[0.38084089]
 [0.34328658]
 [0.39077742]]
Negative and Positive Morality: [[0.25048269]
 [0.26114824]
 [0.32225695]]
On Mending and Ending Things: [[0.24272298]
 [0.31218425]
 [0.39991141]]
A Defence of Rash Vows: [[0.37640904]
 [0.34602045]
 [0.50630658]]
 ```