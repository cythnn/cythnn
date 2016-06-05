import cython
from arch.SkipgramHS cimport SkipgramHS
from tools.ctypes cimport *
from libc.string cimport memset
from tools.blas cimport sdot, saxpy
cimport numpy as np
from numpy import uint64

cdef uLONG rand_prime = 25214903917
cdef int iONE = 1
cdef int iZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0
cdef cREAL fmONE = -1.0

# learns embeddings using cbow against a hierarchical softmax (binary huffmann tree) as output layer
cdef class CbowHS(SkipgramHS):
    def __init__(self, pipeid, learner):
        SkipgramHS.__init__(self, pipeid, learner)

    # process is called with a center position in words (array of word ids), and clower and
    # cupper define valid window boundaries as training context
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper, int length):
        cdef int word, last_word, i, j, inner, exp, l0, l1, wordsprocessed = 0
        cdef cINT *p_inner                                                  # pointers to list of output nodes per wordid
        cdef cBYTE *p_exp                                                   # expected value per output node
        cdef float f                                                        # estimated output
        cdef float g                                                        # gradient
        cdef float cfrac
        cdef float alpha = self.solution.updateAlpha(threadid, 0)           # the current learning rate
        cdef cREAL *hiddenlayer_fw = self.solution.getLayerFw(threadid, 1)  # the hidden layer for feed forward
        cdef cREAL *hiddenlayer_bw = self.solution.getLayerBw(threadid, 1)  # the hidden layer for back propagation

        with nogil:
            for i in range(length):
                word = words[i]
                if cupper[i] > clower[i] + 1:

                    # learn against hier. softmax of the center word
                    p_inner = self.innernodes[word]              # points to list of inner nodes for word is HS
                    p_exp = self.expected[word]                  # points to expected value for inner node (left=0, right=1)

                    # set hidden layer to average of embeddings of the context words
                    memset(hiddenlayer_fw, 0, self.vectorsize * 4)
                    memset(hiddenlayer_bw, 0, self.vectorsize * 4)
                    cfrac = 1.0 / (cupper[i] - clower[i] - 1)
                    for j in range(clower[i], cupper[i]):
                        if i != j:
                            last_word = words[j]
                            l0 = last_word * self.vectorsize
                            saxpy( & self.vectorsize, &cfrac, & self.w0[l0], &iONE, hiddenlayer_fw, &iONE)

                    while True:
                        inner = p_inner[0]        # iterate over the inner nodes, until the root (inner = 0)
                        exp = p_exp[0]

                        # index for last_word in weight matrix w0, inner node in w1
                        l1 = inner * self.vectorsize

                        # energy emitted to inner tree node (output layer)
                        f = sdot( &self.vectorsize, hiddenlayer_fw, &iONE, &self.w1[l1], &iONE)

                        # commonly, when g=0 or g=1 there is nothing to train
                        if f >= -self.MAX_SIGMOID and f <= self.MAX_SIGMOID:
                            # compute the gradient * alpha
                            f = self.sigmoidtable[<int>((f + self.MAX_SIGMOID) * (self.SIGMOID_TABLE / self.MAX_SIGMOID / 2))]
                            g = (1 - exp - f) * alpha

                            # update the inner node (appears only once in a path)
                            # then add update to hidden layer
                            saxpy( &self.vectorsize, &g, &(self.w1[l1]), &iONE, hiddenlayer_bw, &iONE)
                            saxpy( &self.vectorsize, &g, hiddenlayer_fw, &iONE, &(self.w1[l1]), &iONE)

                        # check if we backpropagated against the root (inner=0)
                        if inner == 0:
                            break
                        else:
                            p_inner += 1    # otherwise traverse pointers up the tree to the next inner node
                            p_exp += 1

                    for j in range(clower[i], cupper[i]):
                        if i != j:
                            last_word = words[j]
                            l0 = last_word * self.vectorsize
                            saxpy( &self.vectorsize, &fONE, hiddenlayer_bw, &iONE, &(self.w0[l0]), &iONE)

                # update number of words processed, and alpha every 10k words
                wordsprocessed += 1
                if wordsprocessed > self.updaterate:
                    alpha = self.solution.updateAlpha(threadid, wordsprocessed)
                    wordsprocessed = 0
            self.solution.updateAlpha(threadid, wordsprocessed) # push the remainder of wordsprocessed to progress
