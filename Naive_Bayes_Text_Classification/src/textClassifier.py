#!/usr/bin/python

import re
import nltk
import numpy
from math import log
from numpy.linalg import norm
from nltk import word_tokenize
from nltk import FreqDist as fd
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder

class NaiveBayes:
    # Member variiables of the class
    def __init__(self):
        self.alpha = 0.1
        self.vocabSize = 0
        self.totalDocs = 0
        self.speakerData = {}
        self.totalTrainData = ""
        self.bagOfWords = {}
        self.totalWords = 0
        self.uniqueList = []

    def __readfile(self, fileName):
        data = ""
        try:
            f = open(fileName, "r")
            data = f.readlines()
            f.close()
        except Exception as e:
            print (e)
            exit(-1)
        return data


    def __getSpeaketDict(self):
        for speaker in self.speakerData:
            data = self.speakerData[speaker]['statement'].strip()
            self.totalTrainData += data+" "
            self.speakerData[speaker]["words"] = dict(fd(word_tokenize(data)))
            self.speakerData[speaker]["wordCount"] = sum(self.speakerData[speaker]["words"].values())
            self.speakerData[speaker]["classVocabSize"] = len(self.speakerData[speaker]["words"].keys())
            self.uniqueList.append(self.speakerData[speaker]["words"].keys())


    def __genBagOfWords(self):
        self.bagOfWords = dict(fd(word_tokenize(self.totalTrainData.strip())))
        self.vocabSize = len(self.bagOfWords.keys())
        self.totalwords = sum(self.bagOfWords.values())


    def __prepareData(self, data):
        data = re.sub('[ ]+',' ', re.sub('[.,\?!-]',' ',data.strip()))
        data = re.sub("[']","",data)
        data = data.strip().split(' ')
        return data[0],data[1:]


    def parseTrainingData(self, fileName):
        data = self.__readfile(fileName)
        self.totalDocs = len(data)
        for line in data:
            speakerName, ldata = self.__prepareData(line)
            if self.speakerData.get(speakerName):
                self.speakerData[speakerName]["statement"] += " ".join(ldata)+" "
                self.speakerData[speakerName]["docCount"] += 1
            else:
                self.speakerData[speakerName] = {"statement":"", "docCount":1, "wordCount":0, "words":{}}
                self.speakerData[speakerName]["statement"]= " ".join(ldata)+" "

        self.__getSpeaketDict()
        self.__genBagOfWords()




    def parseTestData(self, fileName):
        testData = self.__readfile(fileName)
        testSpeakersData = {}
        for line in testData:
            speakerName, speakerStatement = self.__prepareData(line)
            if testSpeakersData.get(speakerName):
                testSpeakersData[speakerName].append(" ".join(speakerStatement).strip())
            else:
                testSpeakersData[speakerName] = [" ".join(speakerStatement).strip()]
        return testSpeakersData


    def weightedScore(self, data):
        data = data.strip().split(' ')
        result ={}
        speakers = self.speakerData.keys()
        for speaker in speakers:
            localScore = 0
            classProb = float(self.speakerData[speaker].get("docCount")/float(self.totalDocs))
            for word in data:
                count = self.speakerData[speaker]["words"].get(word, 0)
                if self.uniqueList.count(word) == 1:
                    count *= 2
                else:
                    count /= 2
                localScore += log((count + self.alpha)/((self.alpha*self.vocabSize) + self.speakerData[speaker].get("wordCount")))
            result[localScore + log(classProb)] = speaker
        predictedSpeaker = result[max(result.keys())]
        return predictedSpeaker



    def score(self, data):
        data = data.strip().split(' ')
        result ={}
        speakers = self.speakerData.keys()
        for speaker in speakers:
            localScore = 0
            classProb = float(self.speakerData[speaker].get("docCount")/float(self.totalDocs))
            for word in data:
                count = self.speakerData[speaker]["words"].get(word, 0)
                localScore += log((count + self.alpha)/((self.alpha*self.vocabSize) + self.speakerData[speaker].get("wordCount")))
            result[localScore + log(classProb)] = speaker
        predictedSpeaker = result[max(result.keys())]
        return predictedSpeaker


    def computeModel(self, testData):
        positiveCount = negativeCount = 0
        positiveWCount = negativeWCount = 0
        for actualSpeaker in testData.keys():
            dataToPredict = testData[actualSpeaker]
            for statement in dataToPredict:
                predictedSpeaker = self.score(statement)
                #predictedSpeaker = self.weightedScore(statement)
                if actualSpeaker == predictedSpeaker:
                    positiveCount += 1
                else:
                    negativeCount += 1
                predictedWSpeaker = self.weightedScore(statement)
                if actualSpeaker == predictedWSpeaker:
                    positiveWCount += 1
                else:
                    negativeWCount += 1
        print ("\n\nMultinomial Naive Bayes")
        print ("Positive Count : "+str(positiveCount))
        print ("Negative Count : "+str(negativeCount))
        print ("Accuracy : "+str(positiveCount/float(positiveCount+negativeCount)*100)+"%")
        print ("\n\nWeighted Multinomial Naive Bayes")
        print ("Positive Count : "+str(positiveWCount))
        print ("Negative Count : "+str(negativeWCount))
        print ("Accuracy : "+str(positiveWCount/float(positiveWCount+negativeWCount)*100)+"%")


    def printTrainingModel(self):
        print ("Total Number of Words :  "+str(self.totalwords))
        print ("Vocab Size :  "+str(self.vocabSize) )
        print ("Total number of Docs : "+str(self.totalDocs))
        # print ("Bag of words for the whole training data : " )
        # print(self.bagOfWords)
        print ("Speaker\t:\t\tDocument Count\tWord Count")
        for speaker in self.speakerData:
            print (speaker+"\t\t:\t"+str(self.speakerData[speaker]["docCount"])+"\t:\t"+
                    str(self.speakerData[speaker]["wordCount"]))
        print ("")


    def __generateVectors(self, decisionAttributes, fileName="../data/train"):
        data = self.parseTestData(fileName)
        sparseMatrix = {}
        for speaker in data.keys():
            for statement in data[speaker]:
                newCr = [0]*len(decisionAttributes)
                sList = statement.split(' ')
                for word in sList:
                    if word in decisionAttributes:
                        index = decisionAttributes.index(word)
                        newCr[index] += 1
                if speaker in sparseMatrix.keys():
                    sparseMatrix[speaker].append(newCr)
                else:
                    sparseMatrix[speaker]=[newCr]
        return sparseMatrix


    def __scoreKnn(self, train, test, kneighbours=1):
        print ("Evaluating k-NN")
        convertedCentroids = {}
        for speaker in train.keys():
            convertedCentroids[speaker] = numpy.array(train[speaker])
        positiveCount = negativeCount = 0
        for actualSpeaker in test.keys():
            for vector in test[actualSpeaker]:
                distances = {}
                vector = numpy.array(vector)
                val = 999999999
                for trainSpeaker in convertedCentroids.keys():
                    tempval = norm(vector - convertedCentroids[trainSpeaker])
                    if val > tempval:
                        val = tempval
                        distances[trainSpeaker] = val
                distances = sorted(distances.items(), key=lambda x: x[1])
                #print (distances)
                #print (actualSpeaker+"\t\t\t"+distances[-1][0])
                if actualSpeaker == distances[-1][0]:
                    positiveCount += 1
                else:
                    negativeCount += 1
        print ("Positive Count : "+str(positiveCount))
        print ("Negative Count : "+str(negativeCount))
        print ("Accuracy : "+str(positiveCount/float(positiveCount+negativeCount)*100)+"%")


    def __generateCentroids(self, decisionAttributes, fileName="../data/train"):
        centroids = {}
        for speaker in self.speakerData.keys():
            newCr = [0]*len(decisionAttributes)
            statement = self.speakerData[speaker]['statement']
            statement = statement.strip().split(' ')
            for word in statement:
                if word in decisionAttributes:
                    index = decisionAttributes.index(word)
                    newCr[index] += 1
            centroids[speaker] = newCr
        return centroids


    def kNearestNeighbours(self):
        print ("\n\n\nkNearestNeighbours")
        copyOfBag = dict(self.bagOfWords)
        for word in set(stopwords.words('english')):
            if copyOfBag.get(word):
                del copyOfBag[word]
        cattributes = sorted(copyOfBag.items(), key=lambda x: x[1], reverse=True)[:500]
        trainSparseMatrix = self.__generateVectors(dict(cattributes).keys())
        testSparseMatrix = self.__generateVectors(dict(cattributes).keys(), "../data/test")
        self.__scoreKnn(trainSparseMatrix, testSparseMatrix)


    def bigram(self, testData):
        print ("\n\nBigram Naive Bayes")
        trainBiGram = {}
        posCount = negCount = 0
        totalBigrams = 0
        self.alpha = 1.0

        for speaker in self.speakerData:
            tempdata = BigramCollocationFinder.from_words(self.speakerData[speaker]['statement'].split(), window_size = 2)
            trainBiGram[speaker] = dict(tempdata.ngram_fd.items())
            totalBigrams += len(trainBiGram[speaker].keys())

        for actualSpeaker in testData.keys():
            classProb = float(self.speakerData[speaker].get("docCount")/float(self.totalDocs))
            localScore = 0
            result = {}
            for statement in testData[actualSpeaker]:
                statement = statement.strip().split(' ')
                for i in range(len(statement)-1):
                    count = trainBiGram[speaker].get(tuple([statement[i],statement[i+1]]),
                            self.speakerData[speaker]["words"].get(statement[i],0))
                    localScore += log((count + self.alpha)/((self.alpha*self.vocabSize) + len(trainBiGram[speaker].keys())))
                result[localScore + log(classProb)] = speaker
                predictedSpeaker = result[max(result.keys())]
                if predictedSpeaker == actualSpeaker:
                    posCount += 1
                else:
                    negCount += 1
        print ("Positive Count : "+str(posCount))
        print ("Negative Count : "+str(negCount))
        print ("Accuracy : "+str(posCount/float(posCount+negCount)*100)+"%")


# Main function to build a model and analyse it
def main():
    print ("# Intializing the model")
    nb = NaiveBayes()
    print ("# Building the model using training data ")
    nb.parseTrainingData("../data/train")
    print ("# Parsing the test dataset")
    testData = nb.parseTestData("../data/test")
    print ("# Testing the model using test data ")
    print ("# Predicting test data ")
    nb.computeModel(testData)
    nb.bigram(testData)
    nb.kNearestNeighbours()

# Main starts here
if __name__=="__main__":
    main()

