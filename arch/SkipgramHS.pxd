from pipe.cpipe cimport CPipe
from model.solution cimport *

cdef class SkipgramHS(CPipe):
    cdef cINT **innernodes  # points to the HS tree
    cdef cBYTE **expected
    cdef int vectorsize     # size of the hidden layer (=size of the learned embeddings)

    # convenient lookup table for sigmoid function
    cdef int MAX_SIGMOID, SIGMOID_TABLE
    cdef cREAL *sigmoidtable

    # shared weight matrices w0 and w1 and
    cdef cREAL *w0, *w1

    # when split=1, word2taskid assigns the learning of each word to a task id, to lower memory collisions when learning
    cdef int split
    cdef cINT *word2taskid

    cdef void process(self, int threadid, int taskid, cINT *words, cINT *clower, cINT *cupper, int mode, int length)



