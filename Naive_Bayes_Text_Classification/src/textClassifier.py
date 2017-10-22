import re
import math
import pprint
from nltk import FreqDist as fd
from nltk import word_tokenize

pp = pprint.PrettyPrinter(indent=3)

class NaiveBayes:
    # Member variiables of the class
    vocabSize = 0
    totalDocs = 0
    speakerData = {}
    totalTrainData = ""
    bagOfWords = {}
    totalWords = 0

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


    def score(self, data):
        data = data.strip().split(' ')
        result ={}
        speakers = self.speakerData.keys()
        for speaker in speakers:
            localScore = 0
            classProb = float(self.speakerData[speaker].get("docCount")/float(self.totalDocs))
            for word in data:
                count = self.speakerData[speaker]["words"].get(word, 0)
                localScore += math.log((count + 1.0)/(self.vocabSize + self.speakerData[speaker].get("wordCount")))
            result[localScore + math.log(classProb)] = speaker
        predictedSpeaker = result[max(result.keys())]
        return predictedSpeaker


    def computeModel(self, testData):
        positiveCount = negativeCount = 0
        for actualSpeaker in testData.keys():
            dataToPredict = testData[actualSpeaker]
            for statement in dataToPredict:
                predictedSpeaker = self.score(statement)
                if actualSpeaker == predictedSpeaker:
                    positiveCount += 1
                else:
                    negativeCount += 1
        print ("Positive Count : "+str(positiveCount))
        print ("Negative Count : "+str(negativeCount))
        print ("Accuracy : "+str(positiveCount/float(positiveCount+negativeCount)*100)+"%")


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


if __name__=="__main__":
    main()

