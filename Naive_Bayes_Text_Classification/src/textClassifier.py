import re
import pprint

pp = pprint.PrettyPrinter(indent=3)

def readfile(fileName):
    data = ""
    try:
        f = open(fileName, "r")
        data = f.readlines()
        f.close()
    except Exception as e:
        print (e)
        exit(-1)
    return data


def __getSpeaketDict(speakerData):
    vocabSize = ""
    for speaker in speakerData:
        data = speakerData[speaker]['statement'].strip()
        setWords = set(data.split(' '))
        for word in setWords:
            wc = re.findall(word, data)
            speakerData[speaker]["words"][word] = len(wc)
        vocabSize += data+" "
    return speakerData, len(set(vocabSize.strip().split(' ')))


def __prepareData(data):
    data = re.sub('[ ]+',' ', re.sub('[.,\?]',' ',data.strip()))
    data = data.split(' ')
    return data[0],data[1:]


def parseTrainingData(data):
    speakers = {}
    totalwords = 0
    for line in data:
        # line = re.sub('[ ]+',' ', re.sub('[.,\?]',' ',line.strip()))
        speakerName, ldata = __prepareData(line)
        totalwords += len(ldata)

        if speakers.get(speakerName):
            speakers[speakerName]["statement"] += " ".join(ldata)+" "
            speakers[speakerName]["docCount"] += 1
            speakers[speakerName]["wordCount"] += len(ldata)
        else:
            speakers[speakerName] = {"statement":"", "docCount":1, "wordCount":len(ldata), "words":{}}
            speakers[speakerName]["statement"]= " ".join(ldata)+" "

    speakers, vocabSize = __getSpeaketDict(speakers)
    return speakers, totalwords, vocabSize


def prepareModel(testData):
    for line in testData:
        print (line.strip())


def main():
    trainData = readfile("../data/train")
    testData = readfile("../data/test")

    speakerData, totalwords, vocabSize= parseTrainingData(trainData)

    pp.pprint(speakerData)
    print ("Main total words :  ")
    print(totalwords)
    print ("Main Vocab words :  ")
    print(vocabSize)
    prepareModel(testData)


if __name__=="__main__":
    main()
