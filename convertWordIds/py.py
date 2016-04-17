from pipe.py import pypipe
from numpy import fromiter, int32
from tools.taketime import taketime

# converts the input array of word id's, the input should be a wordstream, which considers the models #windowsize
# and provides wentBack and wentPast to indicate words prepended and appended that do not belong to the
# designated range in a partial file, which needs to be passed for proper processing

class convertWordIds(pypipe):
    def feed(self, input):
        def genWords(str, vocab):
            for term in str:
                word = vocab.get(term)
                if word is not None:
                    yield word.index

        if hasattr(self, 'successor'):
            self.successor.feed( ( fromiter(genWords(input, self.model.vocab), dtype=int32), input.wentBack, input.wentPast ) )
        else:
            for i, w in enumerate(genWords(input, self.model.vocab)):
                if (i % 10000):
                    print(self.threadid, i, w)

