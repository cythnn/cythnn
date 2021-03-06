import cython

from pipe.cpipe cimport CPipe
from numpy import uint64
from tools.ctypes cimport *
from libc.math cimport sqrt
import numpy as np
from numpy cimport *
from libc.stdio cimport *

cdef uLONG rand = uint64(25214903917)

# for an input array of word id's and a given window size, this determines the context that is considered for every
# word position, by giving the lower (inclusive) and upper (exclusive) bounds. Corresponding to the orginal Word2Vec code
# for each position, a symetrical context size is sampled from a uniform distribution from 1 to window_size. The context
# is truncated over sentence boundaries (wordid = 0).
cdef class DownSample(CPipe):
    def __init__(self, pipeid, learner):
        CPipe.__init__(self, pipeid, learner)
        self.downsample = self.model.downsample
        if self.downsample > 0:
            self.random = 1
            self.vocabularysize = len(self.model.vocab)
            self.totalwords = self.model.vocab.totalwords
            self.corpusfrequency = allocInt(self.vocabularysize)
            for i in range(1, self.vocabularysize):
                self.corpusfrequency[i] = self.model.vocab.sorted[i].count

    def transform(self): # remove when downsampling is not used
        if self.downsample > 0 and self.model.quiet == 0:
            print("downsampling %0.6f"%self.downsample)
            return self
        return None

    def feed(self, threadid, task):
        cdef:
            int length = task.length
            int wentback = task.wentback
            int wentpast = task.wentpast

        downsampled = self.process(threadid, toIntArray(task.words), &length, &wentback, &wentpast)
        newtask = task.nextTask()
        newtask.words = task.words
        newtask.wentback = wentback
        newtask.wentpast = wentpast
        newtask.length = length
        self.addTask(newtask)
        self.solution.updateProcessed(downsampled)

    # returns the number of items removed from the stream
    cdef int process(self, int threadid, cINT *words, int *length, int *wentback, int *wentpast):
        cdef:
            int pos, pos_downsampled = 0, word, downsampleback = 0, downsamplelength = 0
            float psampledout

        with nogil:
            # downsample frequent terms
            if self.downsample > 0:
                for pos in range(length[0]):
                    word = words[pos]
                    if word > 0: # don't downsample end of sentence, or negative values which paragraph vector uses as movie ids
                        psampledout = (sqrt(self.corpusfrequency[word] / (self.downsample * self.totalwords)) + 1) * \
                                      (self.downsample * self.totalwords) / self.corpusfrequency[word];
                        self.random = self.random * rand + 11;
                        if psampledout < (self.random & 0xFFFF) / 65536.0:
                            if pos < wentback[0]:
                                downsampleback += 1
                            if pos > length[0] - wentpast[0]:
                                wentpast[0] -= 1
                            downsamplelength += 1
                            continue
                    words[pos_downsampled] = words[pos]
                    pos_downsampled += 1

                wentback[0] -= downsampleback
                length[0] -= downsamplelength
        return downsamplelength
