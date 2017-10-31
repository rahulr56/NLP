import collections
import pprint
import math

pp = pprint.PrettyPrinter(indent=4)


class LaplaceUnigramLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.uniDict = collections.defaultdict(lambda: 0)
        self.total = 0
        self.vocab = 0
        self.train(corpus)
        self.totalSize = self.vocab + self.total

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
        """
        for sentence in corpus.corpus:
            for word in sentence.data:
                self.uniDict[word.word] += 1
        self.total = sum(self.uniDict.values())
        self.vocab = len(self.uniDict.keys())

    def score(self, sentence):
        """
        Takes a list of strings as argument and returns the log-probability of
        the sentence using your language model. Use whatever data you computed
        in train() here.  """
        result = 0.0
        for token in sentence:
            count = self.uniDict.get(token) or 0.0
            result += math.log((count + 1.0)/self.totalSize)
        return result
