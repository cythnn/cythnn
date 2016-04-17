import cython
from pipe.cy cimport cypipe
from w2vHSoftmax.cy import build_hs_tree
from model.cy cimport *
from libc.stdio cimport *
from numpy import int32, uint64
from libc.string cimport memset
from blas.cy cimport sdot, saxpy

import numpy as np
cimport numpy as np

cdef cULONGLONG rand = uint64(25214903917)
cdef int iONE = 1
cdef float fONE = 1.0

setvbuf (stdout, NULL, _IONBF, 0);

# learns embeddings using skipgrams against a hierarchical softmax (binary huffmann tree as output layer)
cdef class trainSkipgramHS(cypipe):
    def __init__(self, threadid, model):
        cypipe.__init__(self, threadid, model)
        self.innernodes = self.modelc.innernodes
        self.expected = self.modelc.exp
        self.threads = model.threads
        self.wordsprocessed = 0 # can remember self state
        self.debug = 0

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
    cdef void process(self, cINT * words, cINT * clower, cINT * cupper, int length) nogil:
        cdef int word, last_word, i, j, inner, exp, l0, l1
        cdef cINT * p_inner
        cdef cBYTE * p_exp
        cdef float f, g
        cdef float alpha = self.modelc.updateAlpha(self.threadid, 0)

        for i in range(length):
            word = words[i]
            for j in range(clower[i], cupper[i]):
                if i != j:
                    last_word = words[j]

                    # initialize hidden layer, to aggregate updates for the current last_word
                    memset(self.hiddenlayer, 0, self.vectorsize * 4)

                    p_inner = self.innernodes[word]  # points to list of inner nodes for word is HS
                    p_exp = self.expected[word]  # points to expected value for inner node (left=0, right=1)
                    while True:
                        inner = p_inner[0]  # iterate over the inner nodes, until the root (inner = 0)
                        exp = p_exp[0]

                        # index for last_word in weight matrix w0, inner node in w1
                        l0 = last_word * self.vectorsize
                        l1 = inner * self.vectorsize

                        # energy emitted to inner tree node (output layer)
                        f = sdot(&self.vectorsize, & self.w0[l0], &iONE, & self.w1[l1], &iONE)

                        # commonly, when g=0 or g=1 there is nothing to train
                        if f >= -self.MAX_SIGMOID and f <= self.MAX_SIGMOID:
                            # compute the gradient * alpha
                            g = self.sigmoidtable[ < int > (
                            (f + self.MAX_SIGMOID) * (self.SIGMOID_TABLE / self.MAX_SIGMOID / 2))]
                            g = (1 - exp - g) * alpha

                            # update the inner node (appears only once in a path)
                            # then add update to hidden layer
                            saxpy(&self.vectorsize, &g, & self.w1[l1], &iONE, self.hiddenlayer, &iONE)
                            saxpy(&self.vectorsize, &g, & self.w0[l0], &iONE, & self.w1[l1], &iONE)

                        # check if we backpropagated against the root (inner=0)
                        # if so this was the last inner node for last_word and the
                        # hidden layer must be updated to the embedding of the last_word
                        if inner == 0:
                            saxpy(&self.vectorsize, &fONE, self.hiddenlayer, &iONE, & self.w0[l0], &iONE)
                            break
                        else:
                            p_inner += 1  # otherwise traverse pointers up the tree to the next inner node
                            p_exp += 1

            # update number of words processed, and alpha every 10k words
            self.wordsprocessed += 1
            if self.wordsprocessed > 10000:
                alpha = self.modelc.updateAlpha(self.threadid, self.wordsprocessed)
                self.wordsprocessed = 0
