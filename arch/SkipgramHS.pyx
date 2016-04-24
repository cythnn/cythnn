import cython

from tools.word2vec import createW2V
from w2vHSoftmax.cy import build_hs_tree
from pipe.cpipe import CPipe
from numpy import uint64
from libc.string cimport memset
from tools.blas cimport sdot, saxpy

import numpy as np
cimport numpy as np

cdef cULONGLONG rand = uint64(25214903917)
cdef int iONE = 1
cdef float fONE = 1.0

# learns embeddings using skipgrams against a hierarchical softmax (binary huffmann tree as output layer)
cdef class SkipgramHS(CPipe):
    def __init__(self, pipeid, learner):
        CPipe.__init__(self, pipeid, learner)
        self.innernodes = self.solution.innernodes      # for every wordid, lists the inner nodes ending with 0 for the root
        self.expected = self.solution.exp               # for every inner node, 0 means left turn, 1 right turn, which is the expected value to learn against

        self.split = self.model.split
        self.word2taskid = self.solution.word2taskid    # for split mode, binds wordids to task ids to minimize memory collisions

        self.vectorsize = self.solution.getLayerSize(1) # size of hidden layer
        self.w0 = self.solution.w[0]                    # the lookup matrix for the word embeddings
        self.w1 = self.solution.w[1]                    # the weight matrix that connects the hidden layer to the output layer

        self.MAX_SIGMOID = self.solution.MAX_SIGMOID    # fast lookup table for sigmoid function
        self.SIGMOID_TABLE = self.solution.SIGMOID_TABLE
        self.sigmoidtable = self.solution.sigmoidtable

    # build is executed before the init
    def build(self):
        build_hs_tree(self.learner, self.model)
        createW2V(self.model, self.model.vocsize, self.model.vocsize - 1)

    def feed(self, threadid, task):
        taskid = task.taskid if task.taskid is not None else 0  # only used in split mode
        self.process(threadid, taskid, toIArray(task.words), toIArray(task.clower), toIArray(task.cupper), self.split, task.length)

    # process is called with a center position in words (array of word ids), and clower and
    # cupper define valid window boundaries as training context

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, int threadid, int taskid, cINT * words, cINT * clower, cINT * cupper,
                            int split, int length):
        cdef int word, last_word, i, j, inner, exp, l0, l1, wordsprocessed = 0
        cdef cINT *p_inner                                                  # pointers to list of output nodes per wordid
        cdef cBYTE *p_exp                                                   # expected value per output node
        cdef float f                                                        # estimated output
        cdef float g                                                        # gradient
        cdef bint updated                                                   # in split mode, 1 indicates the embedding must be updated
        cdef float alpha = self.solution.updateAlpha(threadid, 0)           # learning rate
        cdef cREAL *hiddenlayer = self.solution.getLayerBw(threadid, 1)

        with nogil:
            for i in range(length):                     # go over all words, and use its tree path in the output layer
                word = words[i]
                for j in range(clower[i], cupper[i]):   # for every word, go over its context window
                    if i != j:
                        last_word = words[j]            # the word for which the embedding is trained

                        p_inner = self.innernodes[word] # points to list of inner nodes for word is HS
                        p_exp = self.expected[word]     # points to expected value for inner node (left=0, right=1)

                        updated = 0
                        while True:
                            inner = p_inner[0]  # iterate over the inner nodes, until the root (inner = 0)
                            exp = p_exp[0]      # with its expected value

                            # in split mode, only learn when the words task id matches the threads task id
                            if split == 0 or self.word2taskid[inner] == taskid:
                                if not updated: # initialize hidden layer, to aggregate updates for the current last_word
                                    memset(hiddenlayer, 0, self.vectorsize * 4)
                                    updated = 1

                                # indexes for last_word in weight matrix w0, inner node in w1
                                l0 = last_word * self.vectorsize
                                l1 = inner * self.vectorsize

                                # energy emitted to inner tree node (output layer)
                                f = sdot(&self.vectorsize, &self.w0[l0], &iONE, &self.w1[l1], &iONE)

                                # commonly, when f=0 or f=1 there is nothing to train
                                if f >= -self.MAX_SIGMOID and f <= self.MAX_SIGMOID:
                                    # compute the expected value f and gradient g * alpha
                                    f = self.sigmoidtable[
                                        <int> ((f + self.MAX_SIGMOID) * (self.SIGMOID_TABLE / self.MAX_SIGMOID / 2))]
                                    g = (1 - exp - f) * alpha

                                    # update the inner node (appears only once in a path)
                                    # then add update to hidden layer
                                    saxpy(&self.vectorsize, &g, &self.w1[l1], &iONE, hiddenlayer, &iONE)
                                    saxpy(&self.vectorsize, &g, &self.w0[l0], &iONE, &self.w1[l1], &iONE)

                            # check if we backpropagated against the root (inner=0)
                            if inner == 0:
                                break
                            else:
                                p_inner += 1  # otherwise traverse pointers up the tree to the next inner node
                                p_exp += 1
                                if p_inner[0] < 8:
                                    break

                        if updated:
                            saxpy(&self.vectorsize, &fONE, hiddenlayer, &iONE, & self.w0[l0], &iONE)

                # update number of words processed, and alpha every 10k words
                wordsprocessed += 1
                if wordsprocessed > 10000:
                    alpha = self.solution.updateAlpha(threadid, wordsprocessed)
                    wordsprocessed = 0
            self.solution.updateAlpha(threadid, wordsprocessed)
