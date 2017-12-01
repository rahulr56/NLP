#!/bin/python

import pprint
pp=pprint.PrettyPrinter(indent=3)
enToSpanishDict ={}

def __openFile(fileName):
    try:
        f=open(fileName,"r")
        x=f.readlines()
        f.close()
        return x
    except Exception as e:
        print("error in reading file")
        exit (-1);

def __prepareDict():
    f=open("../data/dictionary.txt")
    data = f.readlines()
    f.close()
    for line in data:
        line=line.strip().lower()
        enToSpanishDict[line.split("|")[0]] = line.split('|')[1:]

def directTranslate(data):
    __prepareDict()
    pp.pprint(enToSpanishDict)
    translatedData = []
    for line in data:
        line = line.lower().strip()
        translatedText=""
        for word in line.split(" "):
            translatedText+=(enToSpanishDict.get(word) or [word])[0]+" "
        translatedData.append(translatedText)



if __name__=="__main__":
    testData = __openFile("../data/test/test.txt")
    directTranslate(testData)
