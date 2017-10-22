import re
import math
import pprint

pp = pprint.PrettyPrinter(indent=3)

class NaiveBayes:
    # Member variiables of the class
    vocabSize = 0
    totalDocs = 0
    speakerData = {}
    totalTrainData = ""
    bagOfWords = {}

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


    def __genClassVocab(self):
        for speaker in self.speakerData.keys():
            data = self.speakerData[speaker].get("statement").strip()
            lVocabSize = len(set(data.split(' ')))
            self.speakerData[speaker]["classVocabSize"] = lVocabSize

    def __getSpeaketDict(self, speakerData):
        vocabSize = ""
        for speaker in speakerData:
            data = speakerData[speaker]['statement'].strip()
            self.totalTrainData += data+" "
            setWords = set(data.split(' '))
            for word in setWords:
                wc = re.findall(word, data)
                speakerData[speaker]["words"][word] = len(wc)
            vocabSize += data+" "
        return speakerData, len(set(vocabSize.strip().split(' ')))


    def __genBagOfWords(self):
        data = set(self.totalTrainData.strip().split(' '))
        for word in data:
            self.bagOfWords[word] = len(re.findall(word, self.totalTrainData))


    def __prepareData(self, data):
        data = re.sub('[ ]+',' ', re.sub('[.,\?]',' ',data.strip()))
        data = data.strip().split(' ')
        return data[0],data[1:]


    def parseTrainingData(self, fileName):
        speakers = {}
        data = self.__readfile(fileName)
        self.totalDocs = len(data)
        self.totalwords = 0
        for line in data:
            # line = re.sub('[ ]+',' ', re.sub('[.,\?]',' ',line.strip()))
            speakerName, ldata = self.__prepareData(line)
            self.totalwords += len(ldata)

            if speakers.get(speakerName):
                speakers[speakerName]["statement"] += " ".join(ldata)+" "
                speakers[speakerName]["docCount"] += 1
                speakers[speakerName]["wordCount"] += len(ldata)
            else:
                speakers[speakerName] = {"statement":"", "docCount":1, "wordCount":len(ldata), "words":{}}
                speakers[speakerName]["statement"]= " ".join(ldata)+" "

        self.speakerData, self.vocabSize = self.__getSpeaketDict(speakers)
        self.__genBagOfWords()
        self.__genClassVocab()


    def parseTestData(self, fileName):
        testData = self.__readfile(fileName)
        testSpeakersData = {}
        for line in testData:
            speakerName, speakerStatement = self.__prepareData(line)
            if testSpeakersData.get(speakerName):
                testSpeakersData[speakerName].append(" ".join(speakerStatement))
            else:
                testSpeakersData[speakerName] = [" ".join(speakerStatement)]
        return testSpeakersData


    def score(self, data):
        result ={}
        speakers = self.speakerData.keys()
        for speaker in speakers:
            localScore = 0
            classProb = float(self.speakerData[speaker].get("docCount")/float(self.totalDocs))
            for i in range(len(data)):
                count = self.speakerData[speaker]["words"].get(data[i]) or 1
                localScore += math.log((count + 1.0)/(self.speakerData[speaker].get("classVocabSize") + self.speakerData[speaker].get("wordCount")))
                # localScore += math.log(count/float(self.speakerData[speaker].get("wordCount")))
            result[localScore + math.log(classProb)] = speaker
        predictedSpeaker = result[max(result.keys())]
        return predictedSpeaker


    def computeModel(self, testData):
        positiveCount = negativeCount = 0
        for actualSpeaker in testData.keys():
            dataToPredict = testData[actualSpeaker]
            for statement in dataToPredict:
                predictedSpeaker = self.score(statement)
                print (actualSpeaker+"\t:\t"+predictedSpeaker)
                if actualSpeaker == predictedSpeaker:
                    positiveCount += 1
                else:
                    negativeCount += 1
        print ("Positive Count : "+str(positiveCount))
        print ("Negative Count : "+str(negativeCount))
        print ("Accuracy : "+str(positiveCount/float(positiveCount+negativeCount)))


    def printTrainingModel(self):
        print ("Total Number of Words :  "+str(self.totalwords))
        print ("Vocab Size :  "+str(self.vocabSize) )
        print ("Total number of Docs : "+str(self.totalDocs))
        print ("Bag of words for the whole training data : " )
        pp.pprint(self.bagOfWords)
        print ("Speakers present in the training data : ")
        pp.pprint(self.speakerData.keys())
        print ("")


def main():
    nb = NaiveBayes()
    nb.parseTrainingData("../data/train")
    testData = nb.parseTestData("../data/test")
    nb.computeModel(testData)


if __name__=="__main__":
    main()

