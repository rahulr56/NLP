import re
import pprint

pp = pprint.PrettyPrinter(indent=3)

class NaiveBayes:
    def readfile(self, fileName):
        data = ""
        try:
            f = open(fileName, "r")
            data = f.readlines()
            f.close()
        except Exception as e:
            print (e)
            exit(-1)
        return data

    def __getSpeaketDict(self, speakerData):
        vocabSize = ""
        for speaker in speakerData:
            data = speakerData[speaker]['statement'].strip()
            setWords = set(data.split(' '))
            for word in setWords:
                wc = re.findall(word, data)
                speakerData[speaker]["words"][word] = len(wc)
            vocabSize += data+" "
        return speakerData, len(set(vocabSize.strip().split(' ')))


    def __prepareData(self, data):
        data = re.sub('[ ]+',' ', re.sub('[.,\?]',' ',data.strip()))
        data = data.strip().split(' ')
        return data[0],data[1:]


    def parseTrainingData(self, data):
        speakers = {}
        totalwords = 0
        for line in data:
            # line = re.sub('[ ]+',' ', re.sub('[.,\?]',' ',line.strip()))
            speakerName, ldata = self.__prepareData(line)
            totalwords += len(ldata)

            if speakers.get(speakerName):
                speakers[speakerName]["statement"] += " ".join(ldata)+" "
                speakers[speakerName]["docCount"] += 1
                speakers[speakerName]["wordCount"] += len(ldata)
            else:
                speakers[speakerName] = {"statement":"", "docCount":1, "wordCount":len(ldata), "words":{}}
                speakers[speakerName]["statement"]= " ".join(ldata)+" "

        speakers, vocabSize = self.__getSpeaketDict(speakers)
        return speakers, totalwords, vocabSize


    def score(self, data):
        pass


    def computeModel(self, testData):
        for line in testData:
            speakerName, speakerStatement = self.__prepareData(line)
            self.score(speakerStatement)


def main():
    nb = NaiveBayes()
    trainData = nb.readfile("../data/train")
    testData = nb.readfile("../data/test")

    speakerData, totalwords, vocabSize= nb.parseTrainingData(trainData)

    pp.pprint(speakerData)
    print ("Main total words :  ")
    print(totalwords)
    print ("Main Vocab words :  ")
    print(vocabSize)
    nb.computeModel(testData)


if __name__=="__main__":
    main()

