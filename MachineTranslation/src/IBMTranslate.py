#!/bin/python

import pprint
from nltk import FreqDist

pp = pprint.PrettyPrinter(indent=2)


class IBMModel:
    def __init__(self):
        self.bigrams = {}
        self.spanishTrain = self.__getData(
            "../data/es-en/train/europarl-v7.es-en.es")
        self.spanishTest = self.__getData("../data/es-en/test/newstest2013.es")
        self.spanishDev = self.__getData("../data/es-en/dev/newstest2012.es")

        self.englishTrain = self.__getData(
            "../data/es-en/train/europarl-v7.es-en.en")
        self.englishTest = self.__getData("../data/es-en/test/newstest2013.en")
        self.englishDev = self.__getData("../data/es-en/dev/newstest2012.en")

        self.englishVocab = set((" ".join(self.englishTrain)).split(" "))
        self.spanishVocab = set((" ".join(self.spanishTrain)).split(" "))
        self.translationPairs = []
        self.englishTotalsDict = {}
        self.foreginUnigram = {}
        self.translationProbabilities = {}

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
        data = f.readlines()
        f.close()
        return data

    def __buildUniGramForeign(self):
        words = (" ".join(self.spanishTrain)).split(' ')
        self.foreginUnigram = dict(FreqDist(words).items())

    def __intiliazeUniformTEF(self):
        self.translationProbabilities = {}
        for foreignWord in self.spanishVocab:
            for englishWord in self.englishVocab:
                self.translationProbabilities[foreignWord][englishWord] = 0.25

    def __genPairs(self):
        for i in range(len(self.englishTrain)):
            self.translationPairs.append((self.spanishTrain[i],
                                          self.englishTrain[i]))

    def __converge(self):
        converged = False
        while not converged:
            for (x, y) in self.translationPairs:
                spanish = x.strip().split(' ')
                english = y.strip().split(' ')
                for eWord in english:
                    self.englishTotalsDict[eWord] = 0
                    for s in spanish:
                        sPorbs = self.translationProbabilities[s]
                        if eWord not in sPorbs:
                            continue

    def buildModel(self):
        self.__genPairs()
        self.__buildUniGramForeign()
        exit(-3)
        self.__intiliazeUniformTEF()
        self.__converge()


def main():
    ibmModel = IBMModel()
    ibmModel.buildModel()
    print(ibmModel)


if __name__ == "__main__":
    main()
