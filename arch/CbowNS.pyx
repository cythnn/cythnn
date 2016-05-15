import cython
from arch.SkipgramNS cimport SkipgramNS
from tools.types cimport *
from libc.string cimport memset
from tools.blas cimport sdot, saxpy

cdef uLONG rand_prime = 25214903917
cdef int iONE = 1
cdef int iZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0
cdef cREAL fmONE = -1.0

# learns embeddings using skipgrams regulating with negative
cdef class CbowNS(SkipgramNS):
    def __init__(self, pipeid, learner):
        SkipgramNS.__init__(self, pipeid, learner)

    # process is called with a center position in words (array of word ids), and clower and
    # cupper define valid window boundaries as training context
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper, int length):
        cdef int word, last_word, i, l0, l1, d, exp, wordsprocessed = 0
        cdef cREAL f, g, cfrac
        cdef float alpha = self.solution.updateAlpha(threadid, 0)
        cdef cREAL *hiddenlayer_fw = self.solution.getLayerFw(threadid, 1)
        cdef cREAL *hiddenlayer_bw = self.solution.getLayerBw(threadid, 1)

        with nogil:
            for i in range(length):
                word = words[i]
                if cupper[i] > clower[i] + 1:

                    #printf("cbow thread %d pos %d lower %d upper %d\n", self.threadid, i, clower[i], cupper[i])
                    # set hidden layer to average of embeddings of the context words
                    memset(hiddenlayer_fw, 0, self.vectorsize * 4)
                    memset(hiddenlayer_bw, 0, self.vectorsize * 4)
                    cfrac = 1.0 / (cupper[i] - clower[i] - 1)
                    for j in range(clower[i], cupper[i]):
                        if i != j:
                            last_word = words[j]
                            l0 = last_word * self.vectorsize
                            saxpy(&self.vectorsize, &cfrac, &self.w0[l0], &iONE, hiddenlayer_fw, &iONE)

                    for d in range(self.negative + 1):
                        if d == 0:
                            word = words[i]
                            exp = 1
                        else:
                            self.random = self.random * rand_prime + 11;  # sample a window size b =[0, windowsize]
                            word = words[ self.random % self.vocabularysize ]
                            exp = 0

                        # index for last_word in weight matrix w0, inner node in w1
                        l1 = word * self.vectorsize

                        # energy emitted to inner tree node (output layer)
                        f = sdot( &self.vectorsize, hiddenlayer_fw, &iONE, &(self.w1[l1]), &iONE)

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
                        saxpy( &self.vectorsize, &g, &(self.w1[l1]), &iONE, hiddenlayer_bw, &iONE)
                        saxpy( &self.vectorsize, &g, hiddenlayer_fw, &iONE, &(self.w1[l1]), &iONE)

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
            self.solution.updateAlpha(threadid, wordsprocessed)
