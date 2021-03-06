import cython
from pipe.cpipe cimport CPipe
from tools.word2vec import createW2V
from numpy import uint64
from libc.string cimport memset
from tools.blas cimport sdot, saxpy
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
cdef class SkipgramNS(CPipe):
    def __init__(self, pipeid, learner):
        CPipe.__init__(self, pipeid, learner)
        self.negative = self.model.negative             # the number of negative samples per word
        self.vocabularysize = len(self.model.vocab)

        self.vectorsize = self.solution.getLayerSize(1) # size of hidden layer
        self.w0 = self.solution.w[0]                    # the lookup matrix for the word embeddings
        self.w1 = self.solution.w[1]                    # the weight matrix that connects the hidden layer to the output layer

        self.MAX_SIGMOID = self.solution.MAX_SIGMOID    # fast lookup table for sigmoid function
        self.SIGMOID_TABLE = self.solution.SIGMOID_TABLE
        self.sigmoidtable = self.solution.sigmoidtable
        self.updaterate = self.model.updaterate
        self.negativesampletablesize = 100000000
        self.negativesampletable = self.buildNegativeSampleTable()
        self.random = allocULong(self.model.threads)
        for i in range(self.model.threads):
            self.random[i] = i

    cdef cINT *buildNegativeSampleTable(self):
        cdef:
            cINT *table = allocInt(self.negativesampletablesize)
            int i, a
            double d1, words_pow = 0

        for i in range(self.vocabularysize):
            words_pow += pow(self.model.vocab.sorted[i].count, 0.75)
        i = 0
        d1 = pow(self.model.vocab.sorted[i].count, 0.75) / words_pow
        for a in range(self.negativesampletablesize):
            table[a] = i
            if (a / <double>self.negativesampletablesize) > d1:
                i += 1
                d1 += pow(self.model.vocab.sorted[i].count, 0.75) / words_pow
        return table

    def build(self):
        createW2V(self.model, self.model.vocsize, self.model.vocsize)

    def feed(self, threadid, task):
        self.process(threadid, toIntArray(task.words), toIntArray(task.clower), toIntArray(task.cupper), task.length)

    # process is called with a center position in words (array of word ids), and clower and
    # cupper define valid window boundaries as training context
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper, int length):
        cdef:
            int word, last_word, i, d, l0, l1, exp, wordsprocessed = 0
            cREAL f, g
            float alpha = self.solution.updateAlpha(threadid, 0)
            cREAL *hiddenlayer = self.solution.getLayerBw(threadid, 1)

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

                                self.random[threadid] = self.random[threadid] * rand_prime + eleven;
                                word = self.negativesampletable[(self.random[threadid] >> 16) % self.negativesampletablesize]
                                if word == 0:
                                    word = self.random[threadid] % (self.vocabularysize - 1) + 1
                                if word == words[i]:
                                    continue
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
                if wordsprocessed > self.updaterate:
                    alpha = self.solution.updateAlpha(threadid, wordsprocessed)
                    wordsprocessed = 0
            self.solution.updateAlpha(threadid, wordsprocessed)

