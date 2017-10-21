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


def parseSpeakerData(data):
    speakers = {}
    totalwords = 0
    for line in data:
        line = re.sub('[ ]+',' ', re.sub('[.,\?]',' ',line.strip()))
        speakerName = line.split(' ')[0]
        ldata = line.split(' ')[1:]
        totalwords += len(ldata)

        if speakers.get(speakerName):
            speakers[speakerName]["statement"] += " ".join(ldata)+" "
            speakers[speakerName]["docCount"] += 1
            speakers[speakerName]["wordCount"] += len(ldata)
        else:
            speakers[speakerName] = {"statement":"", "docCount":1, "wordCount":len(ldata), "words":{}}
            speakers[speakerName]["statement"]= " ".join(ldata)+" "

    speakers, vocabSize = __getSpeaketDict(speakers)
    print("vocabSize :"+str(vocabSize))
    return speakers, totalwords, vocabSize


def main():
    data = readfile("../data/train")
    speakerData, totalwords, vocabSize= parseSpeakerData(data)

    pp.pprint(speakerData)
    print ("Main total words :  ")
    print(totalwords)
    print ("Main Vocab words :  ")
    print(vocabSize)



if __name__=="__main__":
    print("Hello")
    main()

