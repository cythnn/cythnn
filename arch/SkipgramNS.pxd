from pipe.cpipe cimport CPipe
from tools.ctypes cimport *

cdef class SkipgramNS(CPipe):
    cdef:
        int updaterate             # update processed words and alpha after this many processed words
        int vectorsize             # size of the hidden layer (=size of the learned embeddings)
        int vocabularysize         # number of words in the vocabulary
        int negative               # number of negative samples used for regularization
        uLONG* random  # for sampling random numbers

        # convenient lookup table for sigmoid function
        int MAX_SIGMOID, SIGMOID_TABLE
        cREAL *sigmoidtable

        # lookup table for negative samples
        cINT *negativesampletable
        uLONG negativesampletablesize

        # shared weight matrices w0 and w1 and a thread-safe vector for the hidden layer
        cREAL *w0, *w1

        # build a table for negative sampling, cf. the original W2V implementation
        cINT *buildNegativeSampleTable(self)

        void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper, int length)
