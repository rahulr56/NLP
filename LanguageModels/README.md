# Natural Language Processing
## 
## Spell Corrector Using Language Models


### Accuracies:

#### Training data as the actual "holbrook-tagged-train.dat"
```
|---------------------------------------------------|
|    Uniform Language Model:                        |
|    correct: 31 total: 471 accuracy: 0.065817      |
|    Laplace Unigram Language Model:                |
|    correct: 52 total: 471 accuracy: 0.110403      |
|    Laplace Bigram Language Model:                 |
|    correct: 63 total: 471 accuracy: 0.133758      |
|    Stupid Backoff Language Model:                 |
|    correct: 1 total: 471 accuracy: 0.002123       |
|    Custom Language Model:                         |
|    correct: 33 total: 471 accuracy: 0.070064      |
|---------------------------------------------------|
```


#### Training data as the "holbrook-tagged-train.dat + holbrook-tagged-dev.dat"
```
|---------------------------------------------------|
|    Uniform Language Model:                        |
|    correct: 54 total: 471 accuracy: 0.114650      |
|    Laplace Unigram Language Model:                |
|    correct: 111 total: 471 accuracy: 0.235669     |
|    Laplace Bigram Language Model:                 |
|    correct: 340 total: 471 accuracy: 0.721868     |
|    Stupid Backoff Language Model:                 |
|    correct: 340 total: 471 accuracy: 0.721868     |
|    Custom Language Model:                         |
|    correct: 405 total: 471 accuracy: 0.859873     |
|---------------------------------------------------|
```
### Analysis :
* It is very clear that the increase in the training data size caused the test data predition to vary in large amounts. It is also interestin to note that Bigrams always perform better than that of unigrams regardless of the tarining data size. However, it is not true in case of trigrams. Trigrams gave better results when the dataset is huge.
* The same is truw in case of Stupid backoff model. I have used bigram and unigam for the backoff strategy. I am very puzzled to see a decline in the prediction rate for the stupid back off model when the same training and test sets are used. However, it didn't make much of a differnce from that of bigram predictions.
* A small increase in the size of training data caused a huge difference in the accuracy levels for all the language models.


### Problems :
1) Understanging the various files and their structure to design the model is time consuming. A better picture would have been obtained had we implemented everything from the scratch.
2) The training data is too less. So, I suspected my models. Later,after adding the dev data for training, the results were promising.
3) I have downloaded the HolyBrook corpus with more data than train + dev online and fed as input to the program. The program rejected it with some errors. It took me a while to debug them.
4) It is still surprising to see the accuacy of Stupid backoff model declined. I am investigating on the reson still.


### Running the code :
* Clone the repo at [languagemodels_nlp](https://github.com/rahulr56/languagemodels_nlp)
```
$ cd src
$ python3 SpellCorrect.py
```

### Contributor : 
[Rahul Rachapalli](github.com/rahulr56)
