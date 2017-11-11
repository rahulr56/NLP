#!/usr/bin/python

import os
import re
import pprint
import logging
from scipy import spatial
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import urllib2

print ("#Initializing System")
pp = pprint.PrettyPrinter(indent=3)

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

DATAURL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt"
TESTDATA = "./word-test.txt"

data = {}
vocab = {}


class Utils:
    def __returnReqData(self, data, vocabSet):
        reqClasses = ['capital-world', 'currency', 'city-in-state', 'family',
                      'gram1-adjective-to-adverb', 'gram3-comparative',
                      'gram6-nationality-adjective']
        newData = {}
        newVocabSet = {}
        for classes in reqClasses:
            newData[classes] = data[classes]
            newVocabSet[classes] = list(set(vocabSet.get(classes)))
        return newData, newVocabSet

    def getData(self):
        data = ""
        if os.access(TESTDATA, os.R_OK):
            logger.info("Reading test data from file"+TESTDATA)
            data = open(TESTDATA, "r").readlines()
        else:
            logger.info("Reading test data from Web")
            data = urllib2.urlopen(DATAURL).readlines()
        data = [line.strip() for line in data]
        dataDict = {}
        vocabSet = {}
        key = ""
        logger.info("Processing test data")
        for line in data:
            if line[0] == ":":
                key = line[2:]
                dataDict[key] = []
                vocabSet[key] = []
            elif line[0] == '/':
                continue
            else:
                dataDict[key].append((re.sub('[ \t]+', ' ', line)).lower())
                vocabSet[key].append((line.split(' ')[-1]).lower())
        for key in vocabSet.keys():
            vocabSet[key] = list(set(vocabSet.get(key)))
        logger.info("Extracting only required info from test data")
        return self.__returnReqData(dataDict, vocabSet)


def getAccuracies(va, vb, vc, key, model):
    similarity = []
    for d in vocab[key]:
        try:
            vd = model[d]       # numpy.array(model[d])
        except Exception:
            similarity.append(0)
            continue
        similarity.append(1 - spatial.distance.cosine(vd, (vb - va + vc)))
    return vocab[key][min(similarity.index(
        max(similarity)), len(vocab[key])-1)]


def predictData(data, key, model):
    positiveCount = negativeCount = 0
    for line in data:
        va, vb, vc, vdAct = line.split(' ')
        logging.debug("VA : "+va + "___ VB : "+vb+"__VC :"+vc)
        try:
            va, vb, vc = model[va.strip()], \
                model[vb.strip()], model[vc.strip()]
        except Exception:
            negativeCount += 1
            continue
        predicted = getAccuracies(va, vb, vc, key, model)
        logging.debug("Predicted " + predicted + "\t Actual : " + vdAct)
        if vdAct == predicted:
            positiveCount += 1
        else:
            negativeCount += 1
    print ("#######################################")
    print ("#   Class :" + key + "#")
    print ("#  Positive count :" + str(positiveCount))
    print ("#  Negativve count :" + str(negativeCount))
    print ("#  Accuracy : "+str(float(positiveCount) /
                                (positiveCount + negativeCount)))
    print("#######################################")


def buildModels():
    logger.info("Building Google Model")
    googleModel = gensim.models.KeyedVectors.load_word2vec_format(
          './GoogleNews-vectors-negative300.bin', binary=True)
    logger.info("Building Glove Model")
    if not (os.access("glove.840B.300d_out.txt", os.R_OK)):
        glove2word2vec("./glove.840B.300d.txt", "glove.840B.300d_out.txt")
    gloveModel = gensim.models.KeyedVectors.load_word2vec_format(
            './glove.840B.300d_out.txt', binary=False)
    return googleModel, gloveModel


if __name__ == "__main__":
    ut = Utils()
    data, vocab = ut.getData()
    logger.info("Predicting Data")
    googleModel, gloveModel = buildModels()
    print ("------------------------------------------")
    print ("|             Google Model               |")
    print ("------------------------------------------")
    for reqClass in data.keys():
        predictData(data[reqClass], reqClass, googleModel)

    print ("------------------------------------------")
    print ("|              Glove Model               |")
    print ("------------------------------------------")
    for reqClass in data.keys():
        predictData(data[reqClass], reqClass, gloveModel)

    customData = {
                "profession": ["art artist teeth dentist",
                              "eye optician art artist",
                              "physics physicist botany botanist"],
                "attire": ["belt leather tie silk",
                          "cuffs metal shoes rubber",
                          "goggles glass ring platinum"]
		"politicians": ["alexander greece nepolian france", 
				"hindenburg germany tippu mysore",
				"lincoln illinoi trudeau canada"] }
	
    vocab["attire"]=["silk", "rubber", "platinum"]
    vocab["profession"]=["dentist", "artist", "botanist"]
    vocab["politicians"]=["france", "mysore", "canada"]
    logger.info("Prediting custom data")
    print ("\n")
    print ("------------------------------------------")
    print ("|              Custom Data               |")
    print ("------------------------------------------")
    predictData(customData["profession"], "profession", googleModel)
    predictData(customData["profession"], "profession", gloveModel)
    predictData(customData["attire"], "attire", googleModel)
    predictData(customData["politicians"], "politicians", gloveModel)
    predictData(customData["politicians"], "politicians", googleModel)
    predictData(customData["attire"], "attire", gloveModel)
    key = input("Press Q to quit")
        
