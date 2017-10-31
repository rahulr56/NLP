import math
import pprint
from LaplaceBigramLanguageModel import LaplaceBigramLanguageModel

pp = pprint.PrettyPrinter(indent=4)


class StupidBackoffLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.biGramCount = {}
        self.uniGramCount = {}
        self.train(corpus)
        self.vocab = len(self.biGramCount.keys())

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        self.bigram = LaplaceBigramLanguageModel(corpus)
        self.uniGramCount = self.bigram.uniGram.uniDict
        self.biGramCount = self.bigram.bigramCount

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        result = 0.0
        for i in range(1,len(sentence)):
            count = (self.biGramCount.get(sentence[i-1]+" "+sentence[i]) or 0.0)
            if count == 0:
                count =  (self.uniGramCount.get(sentence[i-1]) or 0)
            result += math.log((count + 1.0)/(self.vocab+(self.uniGramCount.get(sentence[i-1]) or 0)))
        return result
