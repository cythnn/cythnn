from pipe.cpipe cimport CPipe
from tools.ctypes cimport *

cdef class SkipgramHS(CPipe):
    cdef:
        cINT **innernodes  # points to the HS tree
        cBYTE **expected
        int vectorsize     # size of the hidden layer (=size of the learned embeddings)
        int vocabularysize

        # convenient lookup table for sigmoid function
        int MAX_SIGMOID, SIGMOID_TABLE
        cREAL *sigmoidtable

        # shared weight matrices w0 and w1
        cREAL *w0, *w1

        int updaterate

        void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper, int length)



