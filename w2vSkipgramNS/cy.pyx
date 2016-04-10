import cython
from pipe.cy cimport cypipe
from w2vHSoftmax.cy import build_hs_tree
from model.cy cimport *

from numpy import int32, uint64
from libc.string cimport memset
from blas.cy cimport sdot, saxpy

import numpy as np
cimport numpy as np

cdef cULONGLONG rand = uint64(25214903917)
cdef int iONE = 1
cdef float fONE = 1.0

# learns embeddings using skipgrams against a hierarchical softmax (binary huffmann tree as output layer)
cdef class trainSkipgramNS(cypipe):
    def __init__(self, threadid, model):
        cypipe.__init__(self, threadid, model)
        self.alpha = model.alpha
        self.negative = model.negative
        self.vocabularysize = len(model.vocab)
        self.random = threadid
        self.wordsprocessed = 0 # can remember self state

        self.vectorsize = self.modelc.getLayerSize(1)
        self.w0 = self.modelc.w[0]
        self.w1 = self.modelc.w[1]
        self.hiddenlayer = self.modelc.getLayer(threadid, 1)

        self.MAX_SIGMOID = self.modelc.MAX_SIGMOID
        self.SIGMOID_TABLE = self.modelc.SIGMOID_TABLE
        self.sigmoidtable = self.modelc.sigmoidtable

    cdef void bindToCypipe(self, cypipe predecessor):
        predecessor.bind(self, <void*>(self.process))

    # no need to implement bind() since skipgram is always last in the pipeline

    # process is called with a center position in words (array of word ids), and clower and
    # cupper define valid window boundaries as training context
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, cINT *words, cINT *clower, cINT *cupper, int length) nogil:
        cdef int word, last_word, i, d, l0, l1, exp
        cdef cREAL f, g

        for i in range(length):

            for j in range(clower[i], cupper[i]):
                if i != j:

                    last_word = words[j]
                    l0 = last_word * self.vectorsize

                    # initialize hidden layer, to aggregate updates for the current last_word
                    memset(self.hiddenlayer, 0, self.vectorsize * 4)

                    # train the target word as a positive sample and #negative samples
                    for d in range(self.negative + 1):
                        if d == 0:
                            word = words[i]
                            exp = 1
                        else:
                            self.random = self.random * rand + 11;  # sample a window size b =[0, windowsize]
                            word = words[ self.random % self.vocabularysize ]
                            exp = 0

                        # index for last_word in weight matrix w0, inner node in w1
                        l1 = word * self.vectorsize

                        # energy emitted to inner tree node (output layer)
                        f = sdot( &self.vectorsize, &(self.w0[l0]), &iONE, &(self.w1[l1]), &iONE)

                        # compute the gradient * alpha
                        if f > self.MAX_SIGMOID:
                            if exp == 1:
                                continue
                            g = -self.alpha
                        elif f < -self.MAX_SIGMOID:
                            if exp == 0:
                                continue
                            g = self.alpha
                        else:
                            g = self.alpha * (exp - self.sigmoidtable[<int>((f + self.MAX_SIGMOID) * (self.SIGMOID_TABLE / self.MAX_SIGMOID / 2))])

                        # update the inner node (appears only once in a path)
                        # then add update to hidden layer
                        saxpy( &self.vectorsize, &g, &(self.w1[l1]), &iONE, self.hiddenlayer, &iONE)
                        saxpy( &self.vectorsize, &g, &(self.w0[l0]), &iONE, &(self.w1[l1]), &iONE)

                    saxpy( &self.vectorsize, &fONE, self.hiddenlayer, &iONE, &(self.w0[l0]), &iONE)

                    # update number of words processed, and alpha every 10k words
            self.wordsprocessed += 1
            if self.wordsprocessed > 10000:
                self.modelc.progress[self.threadid] = self.modelc.progress[self.threadid] + self.wordsprocessed
                self.alpha = self.modelc.alpha * max_float(1.0 - self.modelc.getProgress(), 0.0001)
                self.wordsprocessed = 0
                printf("alpha %d %f\n", self.threadid, self.alpha)