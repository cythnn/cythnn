from pipe.cpipe cimport CPipe
from tools.types cimport *

cdef class SkipgramHS(CPipe):
    cdef cINT **innernodes  # points to the HS tree
    cdef cBYTE **expected
    cdef int vectorsize     # size of the hidden layer (=size of the learned embeddings)

    # convenient lookup table for sigmoid function
    cdef int MAX_SIGMOID, SIGMOID_TABLE
    cdef cREAL *sigmoidtable

    # shared weight matrices w0 and w1
    cdef cREAL *w0, *w1

    cdef int updaterate

    cdef void process(self, int threadid, int taskid, cINT *words, cINT *clower, cINT *cupper, int length)



