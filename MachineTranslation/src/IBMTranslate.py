#!/bin/python
import nltk

from nltk import word_tokenize
from nltk import FreqDist as fd
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder


class IBMModel:
    def __init__(self):
        self.bigrams = {}
        self.spanishTrain = self.__getData("../data/es-en/train/europarl-v7.es-en.es")
        self.spanishTest = self.__getData("../data/es-en/test/newstest2013.es")
        self.spanishDev = self.__getData("../data/es-en/dev/newstest2012.es")

        self.englishTrain = self.__getData("../data/es-en/train/europarl-v7.es-en.en")
        self.englishTest = self.__getData("../data/es-en/test/newstest2013.en")
        self.englishDev = self.__getData("../data/es-en/dev/newstest2012.en")

    def __openFile(self, fileName, mode="r"):
        try:
            f = open(fileName, mode)
            return f
        except Exception as e:
            print("error in readinf file :"+fileName)
            print(e)
            exit(-1)

    def __getData(self, fileName):
        f = self.__openFile(fileName)
        data = f.getlines()
        f.close()
        return data



    def createBigrams(self, data):
        pass

def main():
    ibmModel = IBMModel()
    print(ibmModel)


if __name__ == "__main__":
    main()
