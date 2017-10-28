# Naive Bayes Classification 

### Multinomial Naive Bayes Classifcation 
```
Positive Count : 259
Negative Count : 141
Accuracy : 64.75%
``` 

### Weighted Multinomial Naive Bayes Classification
```
Positive Count : 251
Negative Count : 149
Accuracy : 62.75%
```

### Bigram Naive Bayes
```
Positive Count : 5
Negative Count : 395
Accuracy : 1.25%
```

### k-Nearesr neighbor Prediction 



### Analysis and Implementation 
* I have developed a unigram model to build Naive Bayes classifier.
* As the prediction rate was less for simple naive bayes model, I have implemented add one smoothing to boost the prediction rates. 
* For part 2a, I have implemnted Weighted Niave Bayes classifier with add one smoothing. 
* As a solution for part 2b, I have implemnted bigram model and k-nearest neighbor classifier to predict the speaker which did not produce stisfying results.
* I have plotted top 20 words for each speaker and attached the file `naiveBayes.Rmd` 


### Problems Faced 
1. I have tried Parts of speech tagging. I was not able to make used of the tagged data.
2. I have implemented k-nearest neighbours with k=1 and found that the predictions are all wrong. 
3. Adding multiple neighbors gave a problem of how to predict the speaker for the test data based on the neighbours
4. Bigrams are not giving me a good result. 


### Requirements 
The program is built upon the `numpy` , `nltk` libraries. So, before running the program, install the mentioned requiremnts. 
To install the above librarires using pip, run the following commads:
```
pip install numpy
pip install nltk
```
Note : Provide sudo permissions if necessary.

### Instructions to run 
Clone the repo at [Naivebayes](https://github.com/rahulr56/NLP.git)
Run the following commands 
```
cd NLP/Naive_Bayes_Text_Classification/src
./textClassifier.py
```
#### Sample output:
```
# Intializing the model
# Building the model using training data
# Parsing the test dataset
# Testing the model using test data
# Predicting test data


Multinomial Naive Bayes
Positive Count : 259
Negative Count : 141
Accuracy : 64.75%


Weighted Multinomial Naive Bayes
Positive Count : 251
Negative Count : 149
Accuracy : 62.75%


Bigram Naive Bayes
Positive Count : 5
Negative Count : 395
Accuracy : 1.25%



kNearestNeighbours
Evaluating k-NN
Positive Count : 3
Negative Count : 397
Accuracy : 0.75%
```
![Top 20 word plots for speakers](https://github.com/rahulr56/NLP/blob/master/Naive_Bayes_Text_Classification/src/plot.png)
