import collections
import math
from LaplaceUnigramLanguageModel import LaplaceUnigramLanguageModel


class LaplaceBigramLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        # TODO your code here
        self.bigramCount = collections.defaultdict(lambda: 0)
        self.uniGram = LaplaceUnigramLanguageModel(corpus)
        self.train(corpus)
        self.vocab = len(self.bigramCount.keys())

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        for sentence in corpus.corpus:
            line = sentence.data
            for i in range(1,len(line)):
                self.bigramCount[line[i-1].word+" "+line[i].word] += 1

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        result = 0.0
        for i in range(1,len(sentence)):
            count = self.bigramCount.get(sentence[i-1]+" "+sentence[i]) or 0.0
            result += math.log((count + 1.0)/(self.vocab+(self.uniGram.uniDict.get(sentence[i]) or 0)))
        return result
