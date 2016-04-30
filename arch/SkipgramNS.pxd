from pipe.cpipe cimport CPipe
from tools.types cimport *

cdef class SkipgramNS(CPipe):
    cdef int vectorsize             # size of the hidden layer (=size of the learned embeddings)
    cdef int vocabularysize         # number of words in the vocabulary
    cdef int negative               # number of neagtive samples used for regularization
    cdef unsigned long long random  # for sampling random numbers

    # convenient lookup table for sigmoid function
    cdef int MAX_SIGMOID, SIGMOID_TABLE
    cdef cREAL *sigmoidtable

    # shared weight matrices w0 and w1 and a thread-safe vector for the hidden layer
    cdef cREAL *w0, *w1

    cdef int updaterate

    cdef void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper, int length)
