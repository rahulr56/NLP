#!/bin/python

import copy
import pprint
import os
import pickle as p
from nltk import FreqDist

pp = pprint.PrettyPrinter(indent=2)
modelFile = "../data/ibmModel.pkcl"


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

        self.englishTrain = [line.strip() for line in self.englishTrain]
        self.spanishTrain = [line.strip() for line in self.spanishTrain]
        self.englishVocab = set((" ".join(self.englishTrain)).split(" "))
        self.spanishVocab = set((" ".join(self.spanishTrain)).split(" "))
        self.translationPairs = []
        self.englishTotalsDict = {}
        self.foreginUnigram = {}
        self.t = {}

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
        self.t = dict(zip(list(self.spanishVocab),
                          [{}]*len(self.spanishVocab)))
        self.count = dict(zip(list(self.spanishVocab),
                              [{}] * len(self.spanishVocab)))
        # Initialize the whold dict of t and count with 0s
        print(">>> Initlilize transaltion probabilities and count with 0")
        for (x, y) in self.translationPairs:
            spanish = x.strip().split(' ')
            english = y.strip().split(' ')
            for foreignWord in spanish:
                for englishWord in english:
                    self.t[foreignWord][englishWord] = 0.0
                    self.count[foreignWord][englishWord] = 0.0
        # Normalize t
        print(">>> Normalizing the transaltion probabilities")
        for f in self.t.keys():
            norm = len(self.t[f])
            for e in self.t[f].keys():
                self.t[f][e] = 1/norm

    def __genPairs(self):
        for i in range(len(self.englishTrain)):
            self.translationPairs.append((self.spanishTrain[i],
                                          self.englishTrain[i]))

    def __converge(self):
        tempCount = {}
        oltT = {}
        s_total = {}
        total = {}
        smoothingValue = 1e-12
        limit = 0
        while (oltT != self.t):
            oltT = copy.deepcopy(self.t)
            tempCount = copy.deepcopy(self.count)
            for s in self.spanishVocab:
                total[s] = 0.0
            for (x, y) in self.translationPairs:
                spanish = x.strip().split(' ')
                english = y.strip().split(' ')
                for eWord in english:
                    s_total[eWord] = 0.0
                    for s in spanish:
                        if s not in self.t:
                            continue
                        s_total[eWord] += self.t[s].get(eWord, 0)
                for e in english:
                    for s in spanish:
                        if (s not in tempCount) or (e not in tempCount):
                            continue
                        if (s not in self.t):
                            continue
                        tempCount[s][e] += (self.t[s].get(e, 0) /
                                            (s_total.get(e, smoothingValue) or
                                             smoothingValue))

            self.__updateTranslateProbablilities(tempCount, total)
            limit += 1
            print ("=="*(limit+1)+str(10*(limit+1))+"%")
        with open(modelFile, 'wb') as f:
            p.dump([self.t, tempCount], f)

    def __updateTranslateProbablilities(self, tempCount, total):
        smoothingValue = 1e-12
        for s in self.spanishVocab:
            for e in self.englishVocab:
                if (s not in tempCount) or (e not in tempCount):
                    continue
                if (s not in self.t):
                    continue
                self.t[s][e] = (tempCount[s].get(e, 0) / (total[s] or
                                smoothingValue))

    def buildModel(self):
        print (">>> Initializing the model...")
        self.__genPairs()
        print (">>> Building unigram model from data")
        self.__buildUniGramForeign()
        if not os.path.isfile(modelFile):
            self.__intiliazeUniformTEF()
            print ("Starting the process of converging...")
            self.__converge()
        else:
            print(">>> Loading model from the existing file")
            with open(modelFile, 'rb') as f:
                self.t, self.count = p.load(f)
            print(">>> Modle is loaded succssfully")


def main():
    ibmModel = IBMModel()
    ibmModel.buildModel()
    print(ibmModel)


if __name__ == "__main__":
    main()
