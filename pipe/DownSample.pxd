from pipe.cpipe cimport CPipe
from tools.types cimport *
from numpy cimport *

cdef class DownSample(CPipe):
    cdef uLONG random
    cdef int vocabularysize
    cdef float downsample           # the downsampling parameter
    cdef long totalwords
    cdef cINT* corpusfrequency

    cdef void feed2(self, int threadid, object task, cINT *words, int length, int wentback, int wentpast)

    cdef int process(self, int threadid, cINT *words, int *length, int *wentback, int *wentpast)


