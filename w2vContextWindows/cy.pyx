import cython
from pipe.cy cimport cypipe
from numpy import int32, uint64
from model.cy cimport *
from libc.string cimport memset

import numpy as np
cimport numpy as np

cdef cULONGLONG rand = uint64(25214903917)

# for an input array of word id's and a given window size, this determines the context that is considered for every
# word position, by giving the lower (inclusive) and upper (exclusive) bounds. Corresponding to the orginal Word2Vec code
# for each position, a symetrical context size is sampled from a uniform distribution from 1 to window_size. The context
# is truncated over sentence boundaries (wordid = 0).
cdef class contextWindow(cypipe):
    def __init__(self, threadid, model):
        cypipe.__init__(self, threadid, model)
        self.random = threadid
        self.windowsize = model.windowsize

    cdef void bindToCypipe(self, cypipe predecessor):
        predecessor.bind(self, <void*>self.process)

    cdef void bind(self, cypipe successor, void *method):
        self.successor = successor
        try:
            self.successorMethod = <outputMethod> method
        except:
            raise TypeError("Attempting to bind cypipe_example to a successor cypipe method that does not accept the correct argument types")

    def feed(self, input):
        words, wentback, wentpast = input
        self.feed2process(words, wentback, wentpast)

    cdef void feed2process(self, ndarray wordids, int wentback, int wentpast):
        cdef cINT *words = toIArray(wordids)
        cdef int length = len(wordids)
        with nogil:
            self.process(words, length, wentback, wentpast)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, cINT *words, int length, int wentback, int wentpast) nogil:
        cdef int b, i, pos
        cdef int first_word = 0, next_newline

        cdef cINT *clower = allocI(length)
        cdef cINT *cupper = allocI(length)
        memset(clower, 0, length)
        memset(cupper, 0, length)

        for i in range(wentback):
            if words[i] == 0:                             # sentence boundary
                first_word = i + 1
        pos = wentback                                    # pos is the next word position to try as window center
        while pos < length - wentpast:                   # first_word and next_newline determine sentence boundaries
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
        self.successorMethod(self.successor, words, clower, cupper, length)  # emit the sample
        free(clower)
        free(cupper)

