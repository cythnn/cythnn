import cython
from model.cpipe cimport CPipe
from model.solution cimport *
from numpy import int32, uint64
from libc.string cimport memset
from blas.cy cimport sdot, saxpy
from libc.stdio cimport *

import numpy as np
cimport numpy as np

cdef cULONGLONG rand = uint64(25214903917)
cdef int iONE = 1
cdef float fONE = 1.0

# learns embeddings using skipgrams with negative samples
cdef class SkipgramNS(CPipe):
    def __init__(self, pipeid, learner):
        CPipe.__init__(self, pipeid, learner)
        self.negative = self.model.negative             # the number of negative samples per word
        self.vocabularysize = len(self.model.vocab)
        self.random = 1                                 # used to generate pseudo random numbers

        self.vectorsize = self.solution.getLayerSize(1) # size of hidden layer
        self.w0 = self.solution.w[0]                    # the lookup matrix for the word embeddings
        self.w1 = self.solution.w[1]                    # the weight matrix that connects the hidden layer to the output layer

        self.MAX_SIGMOID = self.solution.MAX_SIGMOID    # fast lookup table for sigmoid function
        self.SIGMOID_TABLE = self.solution.SIGMOID_TABLE
        self.sigmoidtable = self.solution.sigmoidtable

        setvbuf(stdout, NULL, _IONBF, 0);               # for debugging, turn off output buffering

    def feed(self, threadid, task):
        self.process(threadid, toIArray(task.words), toIArray(task.clower), toIArray(task.cupper), task.length)

    # process is called with a center position in words (array of word ids), and clower and
    # cupper define valid window boundaries as training context
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper, int length):
        cdef int word, last_word, i, d, l0, l1, exp, wordsprocessed = 0
        cdef cREAL f, g
        cdef float alpha = self.solution.updateAlpha(threadid, 0)
        cdef cREAL *hiddenlayer = self.solution.getLayerBw(threadid, 1)

        with nogil:
            for i in range(length):

                for j in range(clower[i], cupper[i]):
                    if i != j:

                        last_word = words[j]
                        l0 = last_word * self.vectorsize

                        # initialize hidden layer, to aggregate updates for the current last_word
                        memset(hiddenlayer, 0, self.vectorsize * 4)

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
                            f = sdot( &self.vectorsize, &self.w0[l0], &iONE, &self.w1[l1], &iONE)

                            # compute the gradient * alpha
                            if f > self.MAX_SIGMOID:
                                if exp == 1:
                                    continue
                                g = -alpha
                            elif f < -self.MAX_SIGMOID:
                                if exp == 0:
                                    continue
                                g = alpha
                            else:
                                g = alpha * (exp - self.sigmoidtable[<int>((f + self.MAX_SIGMOID) * (self.SIGMOID_TABLE / self.MAX_SIGMOID / 2))])

                            # update the inner node (appears only once in a path)
                            # then add update to hidden layer
                            saxpy( &self.vectorsize, &g, &self.w1[l1], &iONE, hiddenlayer, &iONE)
                            saxpy( &self.vectorsize, &g, &self.w0[l0], &iONE, &self.w1[l1], &iONE)

                        saxpy( &self.vectorsize, &fONE, hiddenlayer, &iONE, &self.w0[l0], &iONE)

                # update number of words processed, and alpha every 10k words
                wordsprocessed += 1
                if wordsprocessed > 10000:
                    alpha = self.solution.updateAlpha(threadid, wordsprocessed)
                    wordsprocessed = 0
            self.solution.updateAlpha(threadid, wordsprocessed)

