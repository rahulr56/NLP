import math
import collections
from LaplaceBigramLanguageModel import LaplaceBigramLanguageModel

class CustomLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.trigramCount = collections.defaultdict(lambda: 0)
        bg = LaplaceBigramLanguageModel(corpus)
        self.bigramCount = bg.bigramCount
        self.uniGram = bg.uniGram.uniDict
        self.train(corpus)
        self.vocab = len(self.bigramCount.keys())

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        for sentence in corpus.corpus:
            line = sentence.data
            for i in range(1,len(line)-1):
                self.trigramCount[line[i-1].word+" "+line[i].word+" "+line[i+1].word] += 1

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        result = 0.0
        for i in range(1,len(sentence)-1):
            count = (self.trigramCount.get(sentence[i-1]+" "+sentence[i]+" "+sentence[i+1])) or 0
            result += math.log((count + 1.0)/(self.vocab+(self.uniGram.get(sentence[i]) or 0)))
        return result
