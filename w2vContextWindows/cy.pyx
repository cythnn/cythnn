import cython

from model.cpipe cimport CPipe
from model.learner import Task

from numpy import int32, uint64
from model.solution cimport *
from model.model import Model
from libc.string cimport memset
from libc.math cimport sqrt

import numpy as np
cimport numpy as np

cdef cULONGLONG rand = uint64(25214903917)

# for an input array of word id's and a given window size, this determines the context that is considered for every
# word position, by giving the lower (inclusive) and upper (exclusive) bounds. Corresponding to the orginal Word2Vec code
# for each position, a symetrical context size is sampled from a uniform distribution from 1 to window_size. The context
# is truncated over sentence boundaries (wordid = 0).
cdef class contextWindow(CPipe):
    def __init__(self, pipeid, learner):
        CPipe.__init__(self, pipeid, learner)
        #cdef int i
        self.random = 1
        self.windowsize = self.model.windowsize
        self.vocabularysize = len(self.model.vocab)
        self.sample = self.model.sample if hasattr(self.model, 'sample') else 0
        self.totalwords = self.model.vocab.totalwords
        self.debug = 0
        if self.sample > 0:
            self.corpusfrequency = allocI(self.vocabularysize)
            for i in range(1, self.vocabularysize):
                self.corpusfrequency[i] = self.model.vocab.sorted[i].count

    #def feed(self, input):
    def feed(self, threadid, task):
        #print("feed", threadid, task)
        words, wentback, wentpast = task.pyparams
        clower = np.zeros(len(words))
        cupper = np.zeros(len(words))
        if self.debug: print("ContextWindows", threadid, len(words), wentback, wentpast)

        downsampled = self.process(threadid, toIArray(words), toIArray(clower), toIArray(cupper),
                     len(words), wentback, wentpast)
        self.solution.updateProcessed(downsampled)

        split = self.model.split if hasattr(self.model, 'split') else 0
        if split == 1:
            for taskid in range(self.getTaskids(task)):
                newtask = Task(taskid=taskid)
                newtask.words = words
                newtask.clower = clower
                newtask.cupper = cupper
                newtask.length = len(words)
                newtask.priority = 1.0 / len(words)
                self.addTask(newtask, task)
        else:
            newtask = Task()
            newtask.words = words
            newtask.clower = clower
            newtask.cupper = cupper
            newtask.length = len(words)
            newtask.priority = 1.0 / len(words)
            self.addTask(newtask, task)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int process(self, int threadid, cINT *words, cINT *clower, cINT *cupper,
                     int length, int wentback, int wentpast):
        if self.debug: printf("ContextWindows\n")
        cdef int b, i, pos, pos_downsampled = 0, word, downsampleback = 0, downsamplelength = 0
        cdef int first_word = 0, next_newline
        cdef float psampledout

        with nogil:
            # downsample frequent terms
            if self.sample > 0:
                for pos in range(length):
                    word = words[pos]
                    if word != 0: # don't downsample end of sentence
                        psampledout = (sqrt(self.corpusfrequency[word] / (self.sample * self.totalwords)) + 1) * (self.sample * self.totalwords) / self.corpusfrequency[word];
                        self.random = self.random * rand + 11;
                        if psampledout < (self.random & 0xFFFF) / 65536.0:
                            if pos < wentback:
                                downsampleback += 1
                            if pos > length - wentpast:
                                wentpast -= 1
                            downsamplelength += 1
                            continue

                    words[pos_downsampled] = words[pos]
                    pos_downsampled += 1
                wentback -= downsampleback
                length -= downsamplelength

            for i in range(wentback):                         # find the first word in the first sentence
                if words[i] == 0:                             # sentence boundary
                    first_word = i + 1
            pos = wentback                                    # pos is the next word position to try as window center

            while pos < length - wentpast:                    # first_word and next_newline determine sentence boundaries
                next_newline = length
                for i in range(pos, length - wentpast):
                    if words[i] == 0:
                        next_newline = i
                        break;
                if pos < next_newline and next_newline - first_word > 2: # a sentence of size > 1 is found
                    while pos < next_newline and pos < length - wentpast:
                        self.random = self.random * rand + 11;   # sample a window size b =[0, windowsize]
                        b = self.random % self.windowsize
                        clower[pos] = max_int(first_word, pos - self.windowsize + b) # limit the window with sentence bounaries
                        cupper[pos] = min_int(pos + self.windowsize + 1 - b, next_newline)
                        pos += 1
                pos = next_newline + 1
                first_word = pos
            if self.debug:  printf("thread %d ContextWindows 2\n", threadid)
        return downsamplelength
