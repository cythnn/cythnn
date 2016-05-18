import cython
from pipe.cpipe cimport CPipe
from tools.word2vec import createW2V
from numpy import uint64
from libc.string cimport memset
from tools.blas cimport sdot, saxpy, scopy
from libc.math cimport pow
from libc.stdio cimport *

cdef uLONG rand_prime = uint64(25214903917)
cdef uLONG eleven = uint64(11)
cdef int iONE = 1
cdef int iZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0
cdef cREAL fmONE = -1.0

# learns embeddings using skipgrams with negative samples
cdef class SkipgramNScached(SkipgramNS):
    def __init__(self, pipeid, learner):
        SkipgramNS.__init__(self, pipeid, learner)
        self.cachewords = self.model.cachewords            # gives the boundary for caching frequent words
        self.updatecacherate = self.model.updatecacherate  # when (#processed terms) to update cached words, inner nodes
        print("vocsize", self.vocabularysize)

    # process is called with a center position in words (array of word ids), and clower and
    # cupper define valid window boundaries as training context
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper, int length):
        cdef:
            int word, last_word, i, j, d, l0, l1, exp, wordsprocessed = 0
            cREAL f, g
            float alpha = self.solution.updateAlpha(threadid, 0)
            cREAL *hiddenlayer = self.solution.getLayerBw(threadid, 1)
            cREAL **t0, **t1, **o0, **o1
            cBYTE *cachedword


        with nogil:
            # setup the caches for frequently used words
            t1 = allocRealP(self.vocabularysize)
            o1 = allocRealP(self.cachewords)
            t0 = allocRealP(self.vocabularysize)
            o0 = allocRealP(self.cachewords)
            cachedword = allocByteZeros(self.cachewords)
            for i in range(self.cachewords):
                t0[i] = allocReal(self.vectorsize) # t0 transparently points to a cache location for frequent terms
                o0[i] = allocReal(self.vectorsize) # o0 contains the original value when cached, to compute the update
                t1[i] = allocReal(self.vectorsize) # t0 transparently points to a cache location for frequent terms
                o1[i] = allocReal(self.vectorsize) # o0 contains the original value when cached, to compute the update
            for i in range(self.cachewords, self.vocabularysize):
                t0[i] = &self.w0[i * self.vectorsize]   # t0 transparently point to w0 for infrequent terms
                t1[i] = &self.w1[i * self.vectorsize]   # t0 transparently point to w0 for infrequent terms

            for i in range(length):
                for j in range(clower[i], cupper[i]):
                    if i != j:

                        last_word = words[j]
                        l0 = last_word * self.vectorsize

                        if last_word < self.cachewords and not cachedword[last_word]:
                            #printf("cache %d %d\n", threadid, last_word)
                            scopy( & self.vectorsize, & self.w0[l0], & iONE, t0[last_word], & iONE)
                            scopy( & self.vectorsize, t0[last_word], & iONE, o0[last_word], & iONE)
                            scopy( & self.vectorsize, & self.w1[l0], & iONE, t1[last_word], & iONE)
                            scopy( & self.vectorsize, t1[last_word], & iONE, o1[last_word], & iONE)
                            cachedword[last_word] = 1

                        # initialize hidden layer, to aggregate updates for the current last_word
                        memset(hiddenlayer, 0, self.vectorsize * 4)

                        # train the target word as a positive sample and #negative samples
                        for d in range(self.negative + 1):
                            if d == 0:
                                word = words[i]
                                exp = 1
                            else:
                                self.random[threadid] = self.random[threadid] * rand_prime + eleven;
                                word = self.negativesampletable[(self.random[threadid] >> 16) % self.negativesampletablesize]
                                if word == 0:
                                    word = self.random[threadid] % (self.vocabularysize - 1) + 1
                                if word == words[i]:
                                    continue
                                exp = 0

                            # index for last_word in weight matrix w0, inner node in w1
                            l1 = word * self.vectorsize

                            # cache the word if its frequent and not in cache
                            if word < self.cachewords and not cachedword[word]:
                                #printf("cache %d %d\n", threadid, word)
                                scopy( & self.vectorsize, & self.w0[l1], & iONE, t0[word], & iONE)
                                scopy( & self.vectorsize, t0[word], & iONE, o0[word], & iONE)
                                scopy( & self.vectorsize, & self.w1[l1], & iONE, t1[word], & iONE)
                                scopy( & self.vectorsize, t1[word], & iONE, o1[word], & iONE)
                                cachedword[word] = 1

                            # energy emitted to inner tree node (output layer)
                            f = sdot( &self.vectorsize, t0[last_word], &iONE, t1[word], &iONE)

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
                            saxpy( &self.vectorsize, &g, t1[word], &iONE, hiddenlayer, &iONE)
                            saxpy( &self.vectorsize, &g, t0[last_word], &iONE, t1[word], &iONE)

                        saxpy( &self.vectorsize, &fONE, hiddenlayer, &iONE, t0[last_word], &iONE)

                # update cached words, number of words processed, and alpha at the given updaterate(default=10k words)
                if wordsprocessed % self.updatecacherate == 0 or i == length -1:
                    for j in range(self.cachewords):  # update cached most frequent terms
                        if cachedword[j]:
                            #printf("release %d %d %d\n", threadid, wordsprocessed, j)
                            saxpy( & self.vectorsize, & fmONE, o0[j], & iONE, t0[j], & iONE)
                            saxpy( & self.vectorsize, & fONE, t0[j], & iONE, & self.w0[j * self.vectorsize], & iONE)
                            saxpy( & self.vectorsize, & fmONE, o1[j], & iONE, t1[j], & iONE)
                            saxpy( & self.vectorsize, & fONE, t1[j], & iONE, & self.w1[j * self.vectorsize], & iONE)
                            cachedword[j] = 0

                # update number of words processed, and alpha every 10k words
                wordsprocessed += 1
                if wordsprocessed > self.updaterate:
                    alpha = self.solution.updateAlpha(threadid, wordsprocessed)
                    wordsprocessed = 0
            self.solution.updateAlpha(threadid, wordsprocessed)

            # free memory
            for j in range(self.cachewords):
                free(o0[j]); free(t0[j]);free(o1[j]); free(t1[j]);
            free(o0); free(t0);free(o1); free(t1)
