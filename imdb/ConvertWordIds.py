import re
from numpy import fromiter, int32

from model.learner import Task
from pipe.pipe import Pipe


# converts the input array of word id's, the input should be a wordstream, which considers the models #windowsize
# and provides wentBack and wentPast to indicate words prepended and appended that do not belong to the
# designated range in a partial file, which needs to be passed for proper processing

class convertWordIds(Pipe):
    def feed(self, threadid, task):
        def genWords(input, vocab, idvocab):
            for term in input:
                if term[0] == '#':                      # new imdbid
                    #print("imdbid", int(term[1:]), -1 - idvocab[int(term[1:])])
                    yield -1 - idvocab[int(term[1:])]    # imdb id's are added as negative numbers
                else:
                    word = vocab.get(term)
                    if word is not None:
                        yield word.index

        wordstream = task.inputstream
        wordidstream = (fromiter(genWords(wordstream, self.model.vocab, self.model.indexeditems), dtype=int32))
        newtask = task.nextTask()
        #task.priority=1/len(wordidstream)
        newtask.wentback = 0
        newtask.wentpast = 0
        newtask.words = wordidstream
        newtask.length = len(wordidstream)
        self.addTask(newtask)


