# Vector Semantics
## Word Embeddings

### Using word2vec
Sample Output:
```
#######################################
#  Class :gram3-comparative#
#  Positive count :1131
#  Negative count :201
#  Accuracy : 0.849099099099
#######################################
#  Class :city-in-state#
#  Positive count :0
#  Negative count :2467
#  Accuracy : 0.0
#######################################
#  Class :family#
#  Positive count :399
#  Negative count :107
#  Accuracy : 0.788537549407
#######################################
#  Class :currency#
#  Positive count :1
#  Negative count :865
#  Accuracy : 0.00115473441109
#######################################
#  Class :gram6-nationality-adjective#
#  Positive count :62
#  Negative count :1537
#  Accuracy : 0.0387742338962
#######################################
#  Class :capital-world#
#  Positive count :12
#  Negative count :4512
#  Accuracy : 0.0026525198939
#######################################
#  Class :gram1-adjective-to-adverb#
#  Positive count :506
#  Negative count :486
#  Accuracy : 0.510080645161
#######################################
```


### Using Glove's pretrained 840B x 300d vectors
Sample Output:
```
#######################################
#  Class :gram3-comparative#
#  Positive count :1184
#  Negative count :148
#  Accuracy : 0.888888888889
#######################################
#  Class :city-in-state#
#  Positive count :1000
#  Negative count :1467
#  Accuracy : 0.405350628293
#######################################
#  Class :family#
#  Positive count :488
#  Negative count :18
#  Accuracy : 0.96442687747
#######################################
#  Class :currency#
#  Positive count :24
#  Negative count :842
#  Accuracy : 0.0277136258661
#######################################
#  Class :gram6-nationality-adjective#
#  Positive count :991
#  Negative count :608
#  Accuracy : 0.61976235147
#######################################
#  Class :capital-world#
#  Positive count :3094
#  Negative count :1430
#  Accuracy : 0.683908045977
#######################################
#  Class :gram1-adjective-to-adverb#
#  Positive count :695
#  Negative count :297
#  Accuracy : 0.70060483871
#######################################
```

## Reason for Antonyms appearing colose to one another
Word embeddings are usually trained on an objective that ensures that words occurring in similar contexts have similar embeddings. Antonyms are often interchangeable in context and thus have similar word embeddings even though they denote opposites. If we think of word embeddings as members of a commutative group, then antonyms should be inverses of (as opposed to similar to) each other. Word embeddings only take into consideration the context of when the words occur. They don’t deal with the meanings of the sentences; hence it doesn’t differentiate between antonyms or any other kind of word groups.

