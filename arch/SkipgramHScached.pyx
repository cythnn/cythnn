import cython

from arch.SkipgramHS import SkipgramHS
from libc.string cimport memset
from tools.blas cimport sdot, saxpy, scopy

cdef int iONE = 1
cdef int iZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0
cdef cREAL fmONE = -1.0

# extends SkipgramHS by training the top-#cacheinner inner nodes in a local caches that is updated to
# shared memory every #updatecacherate trained words
cdef class SkipgramHScached(SkipgramHS):
    def __init__(self, pipeid, learner):
        SkipgramHS.__init__(self, pipeid, learner)
        self.cachewords = self.model.cachewords           # gives the boundary for caching frequent words
        self.cacheinner = self.model.cacheinner         # gives the boundary for caching frequent inner nodes
        self.updatecacherate = self.model.updatecacherate  # when (#processed terms) to update cached words, inner nodes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, int threadid, cINT * words, cINT * clower, cINT * cupper, int length):
        cdef:
            int word, last_word, i, j, inner, exp, wordsprocessed = 0
            cINT *p_inner                                                  # pointers to list of output nodes per wordid
            cBYTE *p_exp                                                   # expected value per output node
            float f                                                        # estimated output
            float g                                                        # gradient
            float alpha = self.solution.updateAlpha(threadid, 0)           # learning rate
            cREAL *hiddenlayer = self.solution.getLayerBw(threadid, 1)

            # setup the caches for frequently used words and inner nodes
            cREAL **t0 = allocRealP(self.model.vocsize)
            cREAL **t1 = allocRealP(self.model.vocsize)
            cREAL **o0 = allocRealP(self.cachewords)
            cREAL **o1 = allocRealP(self.cacheinner)
            cBYTE * cachedword = allocByteZeros(self.cachewords)
            cBYTE * cachedinner = allocByteZeros(self.cacheinner)

        with nogil:
            for i in range(self.cachewords):
                t0[i] = allocReal(self.vectorsize)  # t0 transparently points to a cache location for frequent terms
                o0[i] = allocReal(self.vectorsize)  # o0 contains the original value when cached, to compute the update
            for i in range(self.cacheinner):
                t1[i] = allocReal(self.vectorsize)  # t1 transparently points to a cache location for frequent inner nodes
                o1[i] = allocReal(self.vectorsize)  # o1 contains the original vaue when cacahed, to compute the update
            for i in range(self.cacheinner, self.vocabularysize - 1):
                t1[i] = & self.w1[i * self.vectorsize]  # t1 transparently points to w1 for infrequent inner nodes
            for i in range(self.cachewords, self.vocabularysize):
                t0[i] = & self.w0[i * self.vectorsize]  # t0 transparently point to w0 for infrequent terms

            for i in range(length):                     # go over all words, and use its tree path in the output layer
                word = words[i]                         # next center word, whose huffmann tree location is used to learn te context words against
                for j in range(clower[i], cupper[i]):   # for every word, go over its context window
                    if i != j:
                        last_word = words[j]            # the word for which the embedding is trained

                        # initialize hidden layer, to aggregate updates for the current last_word
                        memset(hiddenlayer, 0, self.vectorsize * 4)

                        # cache the word if its frequent and not in cache
                        if last_word < self.cachewords and not cachedword[last_word]:
                            scopy( &self.vectorsize, &self.w0[last_word * self.vectorsize], & iONE, t0[last_word], & iONE)
                            scopy( &self.vectorsize, t0[last_word], &iONE, o0[last_word], &iONE)
                            cachedword[last_word] = 1

                        p_inner = self.innernodes[word] # points to list of inner nodes for word is HS
                        p_exp = self.expected[word]     # points to expected value for inner node (left=0, right=1)

                        while True:

                            inner = p_inner[0]  # iterate over the inner nodes, terminated by -1
                            exp = p_exp[0]      # with its expected value

                            # cache the inner node if its frequent and not in cache
                            if inner < self.cacheinner and not cachedinner[inner]:
                                scopy( &self.vectorsize, &self.w1[inner * self.vectorsize], &iONE, t1[inner], &iONE)
                                scopy( &self.vectorsize, t1[inner], &iONE, o1[inner], &iONE)
                                cachedinner[inner] = 1

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


                # update cached words, number of words processed, and alpha at the given updaterate(default=10k words)
                if wordsprocessed % self.updatecacherate == 0 or i == length -1:
                    for j in range(self.cachewords):  # update cached most frequent terms
                        if cachedword[j]:
                            saxpy( & self.vectorsize, & fmONE, o0[j], & iONE, t0[j], & iONE)
                            saxpy( & self.vectorsize, & fONE, t0[j], & iONE, & self.w0[j * self.vectorsize], & iONE)
                            cachedword[j] = 0
                    for j in range(self.cacheinner):  # update cached most frequent inner nodes
                        if cachedinner[j]:
                            saxpy( & self.vectorsize, & fmONE, o1[j], & iONE, t1[j], & iONE)
                            saxpy( & self.vectorsize, & fONE, t1[j], & iONE, & self.w1[j * self.vectorsize], & iONE)
                            cachedinner[j] = 0

                wordsprocessed += 1
                if wordsprocessed > self.updaterate:
                    alpha = self.solution.updateAlpha(threadid, wordsprocessed)  # update words processd and learning rate alpha
                    wordsprocessed = 0

            alpha = self.solution.updateAlpha(threadid, wordsprocessed)

            # free memory
            for j in range(self.cachewords):
                free(o0[j]); free(t0[j])
            for j in range(self.cacheinner):
                free(o1[j]); free(t1[j])
            free(o0); free(o1); free(t0); free(t1)
