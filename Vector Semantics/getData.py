#!/usr/bin/python

import pprint
import urllib2

pp = pprint.PrettyPrinter(indent=3)

DATAURL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt"


class Utils:
    def __returnReqData(self, data):
        reqClasses = ['capital-world', 'currency', 'city-in-state', 'family',
                      'gram1-adjective-to-adverb', 'gram3-comparative',
                      'gram6-nationality-adjective']
        newData = {}
        for classes in reqClasses:
            newData[classes] = data[classes]
        return newData

    def getData(self):
        data = urllib2.urlopen(DATAURL).readlines()
        data = [line.strip() for line in data]
        dataDict = {}
        key = ""
        for line in data:
            if line[0] == ":":
                key = line[2:]
                dataDict[key] = []
            elif line[0] == '/':
                continue
            else:
                dataDict[key].append(line)
        return self.__returnReqData(dataDict)


if __name__ == "__main__":
    ut = Utils()
    pp.pprint(ut.getData().keys())

