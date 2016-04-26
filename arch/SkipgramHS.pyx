import cython

from tools.word2vec import createW2V
from tools.hsoftmax import hsoftmax
from pipe.cpipe import CPipe
from numpy import uint64
from libc.string cimport memset
from tools.blas cimport sdot, saxpy, scopy

import numpy
cimport numpy

cdef cULONGLONG rand = uint64(25214903917)
cdef int iONE = 1
cdef float fmONE = -1.0
cdef float fONE = 1.0

# learns embeddings using skipgrams against a hierarchical softmax (binary huffmann tree as output layer)
cdef class SkipgramHS(CPipe):
    def __init__(self, pipeid, learner):
        CPipe.__init__(self, pipeid, learner)
        self.innernodes = self.solution.innernodes      # for every wordid, lists the inner nodes ending with 0 for the root
        self.expected = self.solution.exp               # for every inner node, 0 means left turn, 1 right turn, which is the expected value to learn against

        self.vectorsize = self.solution.getLayerSize(1) # size of hidden layer
        self.w0 = self.solution.w[0]                    # the lookup matrix for the word embeddings
        self.w1 = self.solution.w[1]                    # the weight matrix that connects the hidden layer to the output layer

        self.MAX_SIGMOID = self.solution.MAX_SIGMOID    # fast lookup table for sigmoid function
        self.SIGMOID_TABLE = self.solution.SIGMOID_TABLE
        self.sigmoidtable = self.solution.sigmoidtable
        self.wordcache = self.model.wordcache           # gives the boundary for caching frequent words
        self.innercache = self.model.innercache         # gives the boundary for caching frequent inner nodes
        self.updaterate = self.model.updaterate         # when (#processed terms) to update cahed words, inner nodes, processed terms and alpha

    # build is executed before the init
    def build(self):
        hsoftmax(self.learner, self.model)
        createW2V(self.model, self.model.vocsize, self.model.vocsize - 1)

    def feed(self, threadid, task):
        taskid = task.taskid if task.taskid is not None else 0  # only used in split mode
        self.process(threadid, taskid, toIArray(task.words), toIArray(task.clower), toIArray(task.cupper), task.length)

    # process is called with a center position in words (array of word ids), and clower and
    # cupper define valid window boundaries as training context

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, int threadid, int taskid, cINT * words, cINT * clower, cINT * cupper, int length):
        cdef int word, last_word, i, j, inner, exp, wordsprocessed = 0
        cdef cINT *p_inner                                                  # pointers to list of output nodes per wordid
        cdef cBYTE *p_exp                                                   # expected value per output node
        cdef float f                                                        # estimated output
        cdef float g                                                        # gradient
        cdef bint updated                                                   # in split mode, 1 indicates the embedding must be updated
        cdef float alpha = self.solution.updateAlpha(threadid, 0)           # learning rate
        cdef cREAL *hiddenlayer = self.solution.getLayerBw(threadid, 1)

        # setup the caches for frequently used words and inner nodes
        cdef cREAL **t0 = allocRP(self.model.vocsize)
        cdef cREAL **t1 = allocRP(self.model.vocsize)
        cdef cREAL **o0 = allocRP(self.wordcache)
        cdef cREAL **o1 = allocRP(self.innercache)
        cdef cBYTE *updated0 = allocBZeros(self.wordcache)
        cdef cBYTE *updated1 = allocBZeros(self.innercache)
        for i in range(self.wordcache):
            t0[i] = allocR(self.vectorsize) # t0 transparently points to a cache location for frequent terms
            o0[i] = allocR(self.vectorsize) # o0 contains the original value when cached, to compute the update
        for i in range(self.innercache):
            t1[i] = allocR(self.vectorsize) # t1 transparently points to a cache location for frequent inner nodes
            o1[i] = allocR(self.vectorsize) # o1 contains the original vaue when cacahed, to compute the update
        for i in range(self.innercache, self.model.vocsize - 1):
            t1[i] = &self.w1[i * self.vectorsize]   # t1 transparently points to w1 for infrequent inner nodes
        for i in range(self.wordcache, self.model.vocsize):
            t0[i] = &self.w0[i * self.vectorsize]   # t0 transparently point to w0 for infrequent terms

        with nogil:
            for i in range(length):                     # go over all words, and use its tree path in the output layer

                # update cached words, number of words processed, and alpha at the given updaterate(default=10k words)
                if wordsprocessed == 0 or wordsprocessed > self.updaterate:
                    if wordsprocessed > 0:
                        for j in range(self.wordcache): # update cached most frequent terms
                            if updated0[j]:
                                saxpy(&self.vectorsize, &fmONE, o0[j], &iONE, t0[j], &iONE)
                                saxpy(&self.vectorsize, &fONE, t0[j], &iONE, &self.w0[j * self.vectorsize], &iONE)
                                updated0[j] = 0
                        for j in range(self.innercache): # update cached most frequent inner nodes
                            if updated1[j]:
                                saxpy(&self.vectorsize, &fmONE, o1[j], &iONE, t1[j], &iONE)
                                saxpy(&self.vectorsize, &fONE, t1[j], &iONE, &self.w1[j * self.vectorsize], &iONE)
                                updated1[j] = 0
                    alpha = self.solution.updateAlpha(threadid, wordsprocessed) # update words processd and learning rate alpha
                    wordsprocessed = 0

                wordsprocessed += 1
                word = words[i]                         # next center word, whose huffmann tree location is used to learn te context words against

                for j in range(clower[i], cupper[i]):   # for every word, go over its context window
                    if i != j:
                        last_word = words[j]            # the word for which the embedding is trained

                        # initialize hidden layer, to aggregate updates for the current last_word
                        memset(hiddenlayer, 0, self.vectorsize * 4)

                        # cache the word if its frequent and not in cache
                        if last_word < self.wordcache and not updated0[last_word]:
                            scopy( &self.vectorsize, &self.w0[last_word * self.vectorsize], & iONE, t0[last_word], & iONE)
                            scopy( &self.vectorsize, t0[last_word], &iONE, o0[last_word], &iONE)
                            updated0[last_word] = 1

                        p_inner = self.innernodes[word] # points to list of inner nodes for word is HS
                        p_exp = self.expected[word]     # points to expected value for inner node (left=0, right=1)

                        while True:

                            inner = p_inner[0]  # iterate over the inner nodes, terminated by -1
                            exp = p_exp[0]      # with its expected value

                            # cache the inner node if its frequent and not in cache
                            if inner < self.innercache and not updated1[inner]:
                                scopy( &self.vectorsize, &self.w1[inner * self.vectorsize], &iONE, t1[inner], &iONE)
                                scopy( &self.vectorsize, t1[inner], &iONE, o1[inner], &iONE)
                                updated1[inner] = 1

                            # energy emitted to inner tree node (output layer)
                            f = sdot(&self.vectorsize, t0[last_word], &iONE, t1[inner], &iONE)

                            # commonly, when f=0 or f=1 there is nothing to train
                            if f >= -self.MAX_SIGMOID and f <= self.MAX_SIGMOID:
                                # compute the expected value f and gradient g * alpha
                                f = self.sigmoidtable[
                                    <int> ((f + self.MAX_SIGMOID) * (self.SIGMOID_TABLE / self.MAX_SIGMOID / 2))]
                                g = (1 - exp - f) * alpha

                                # update the inner node (appears only once in a path)
                                # then add update to hidden layer
                                saxpy(&self.vectorsize, &g, t1[inner], &iONE, hiddenlayer, &iONE)
                                saxpy(&self.vectorsize, &g, t0[last_word], &iONE, t1[inner], &iONE)

                            # check if we backpropagated against the root (inner=0)
                            p_inner += 1  # otherwise traverse pointers up the tree to the next inner node
                            p_exp += 1
                            if inner == 0: # the root=0 is always last
                                break

                        saxpy(&self.vectorsize, &fONE, hiddenlayer, &iONE, t0[last_word], &iONE)

            for j in range(self.wordcache): # update the cached frequent words
                if updated0[j]:
                    saxpy( & self.vectorsize, & fmONE, o0[j], & iONE, t0[j], & iONE)
                    saxpy( & self.vectorsize, & fONE, t0[j], & iONE, & self.w0[j * self.vectorsize], & iONE)
                free(o0[j]); free(t0[j])
            for j in range(self.innercache): # update the cached frequent inner nodes
                if updated1[j]:
                    saxpy( & self.vectorsize, & fmONE, o1[j], & iONE, t1[j], & iONE)
                    saxpy( & self.vectorsize, & fONE, t1[j], & iONE, & self.w1[j * self.vectorsize], & iONE)
                free(o1[j]); free(t1[j])
            free(o0); free(o1); free(t0); free(t1)
            self.solution.updateAlpha(threadid, wordsprocessed) # update words processed
