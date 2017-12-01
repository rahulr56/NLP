#!/bin/python

import nltk
import pprint
pp = pprint.PrettyPrinter(indent=3)
enToSpanishDict = {}
translatedData = []


class Rules:
    def __init__(self):
        self.data = ""

    # Check if the word is an adjective
    def __checkadjective(self, tupleTag):
        return tupleTag[1] == 'JJ'

    def __checkNoun(self, tupleTag):
        return tupleTag[1] in ["NN", "NNS", "NNP", "NNPS"]

    def __checkNonProperNoun(self, tupleTag):
        return tupleTag[1] in ["NN", "NNS"]

    def __checkProperNoun(self, tupleTag):
        return tupleTag[1] in ["NNP", "NNPS"]

    def __checkVerb(self, tupleTag):
        return tupleTag[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

    def __checkAdverb(self, tupleTag):
        return tupleTag[1] == "RB"

    def __checkVowel(self, letter):
        return letter in ["a", "e", "i", "o", "u"]

    # Rule 1: There should not be same words consecutie to eachother
    def consecutiveRemoval(self, data):
        consecutiveWords = []
        consecutiveWords.append(data[0])
        for i in range(1, len(data)):
            current = data[i]
            previous = data[i-1]
            if current[0] != previous[0]:
                consecutiveWords.append(current)
        return consecutiveWords

    # Rule 2: Adjectives appear before the noun
    def adjectiveNoun(self, data):
        for i in range(1, len(data)):
            current = data[i]
            previous = data[i-1]
            if self.__checkadjective(current) and self.__checkNoun(previous):
                data[i-1] = current
                data[i] = previous
        return data

    # Rule 3: Remove articles before proper nouns
    def articleRemover(self, data):
        articlesRemoved = []
        for i in range(0, len(data)):
            current = data[i]
            if current[1] == 'DT' and i != len(data)-1:
                next = data[i+1]
                if self.__checkProperNoun(next) is False:
                    articlesRemoved.append(current)
            else:
                articlesRemoved.append(current)
        return articlesRemoved

    # Rule 4: Verbs appear before adverbs
    def verbAdverbRelator(self, data):
        reordered = []
        i = 0
        while i < len(data):
            currTuple = data[i]
            currWord = currTuple[0]
            if self.__checkVerb(currTuple):
                nextTuple = data[i+1]
                nextWord = nextTuple[0]
                if self.__checkAdverb(nextTuple):
                    print (currWord + ' ' + nextWord)
                    reordered.append(nextTuple)
                    reordered.append(currTuple)
                    i += 1
                else:
                    reordered.append(currTuple)
            else:
                reordered.append(data[i])
            i += 1
        reordered.append(data[len(data)-1])
        return reordered

    # Rule 5: Remove be before a verb after translation
    def BERemover(self, data):
        removeBe = []
        for index in range(0, len(data)):
            current = data[index]
            if current[0] == 'be' and index != len(data)-1:
                next = data[index+1]
                if self.__checkVerb(next) is False:
                    removeBe.append(current)
            else:
                removeBe.append(current)
        return removeBe

    # RULE 6: Object follows the verb
    def verbObjectRelator(self, data):
        reordered = []
        i = 0
        while i < len(data):
            currTuple = data[i]
            currWord = currTuple[0]
            if currTuple[1] == 'PRP':
                nextTuple = data[i+1]
                nextWord = nextTuple[0]
                if self.__checkVerb(nextTuple):
                    print (currWord + ' ' + nextWord)
                    reordered.append(nextTuple)
                    reordered.append(currTuple)
                    i += 1
            else:
                reordered.append(data[i])
            i += 1
        return reordered

    def printData(self, data):
        line = ""
        for x in data:
            line += x[0] + " "
        print(line)

    def applyRules(self, data):
        print("Appllying Rule 1")
        data = self.consecutiveRemoval(data)
        print("Appllying Rule 2")
        data = self.adjectiveNoun(data)
        print("Appllying Rule 3")
        data = self.articleRemover(data)
        print("Appllying Rule 4")
        data = self.verbAdverbRelator(data)
        print("Appllying Rule 5")
        data = self.BERemover(data)
        print("Appllying Rule 6")
        data = self.verbObjectRelator(data)
        print("PRinting data")
        self.printData(data)


def __openFile(fileName):
    try:
        f = open(fileName, "r")
        x = f.readlines()
        f.close()
        return x
    except Exception as e:
        print("error in reading file")
        print(e)
        exit(-1)


def __prepareDict():
    f = open("../data/directTranslation/dictionary.txt")
    data = f.readlines()
    f.close()
    for line in data:
        line = line.strip().lower()
        enToSpanishDict[line.split("|")[0]] = line.split('|')[1:]


def directTranslate(data):
    __prepareDict()
    for line in data:
        line = line.lower().strip()
        translatedText = ""
        for word in line.split(" "):
            translatedText += (enToSpanishDict.get(word) or [word])[0]+" "
        print("Spanish : " + line)
        print("Translated : " + translatedText+"\n")
        translatedData.append(translatedText)


def posTagging():
    posTagList = []
    ruleApplier = Rules()
    for line in translatedData:
        lines = nltk.word_tokenize(line)
        tags = nltk.pos_tag(lines)
        posTagList.append(tags)
        ruleApplier.applyRules(tags)


if __name__ == "__main__":
    testData = __openFile("../data/directTranslation/test/test.txt")
    directTranslate(testData)
    posTagging()
