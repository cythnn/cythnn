from pipe.cpipe cimport CPipe
from model.solution cimport *       # defines the cREAL and cINT types
from numpy cimport *

cdef class DownSample(CPipe):
    cdef cULONGLONG random
    cdef int vocabularysize
    cdef float downsample           # the downsampling parameter
    cdef long totalwords
    cdef cINT* corpusfrequency

    cdef void feed2(self, int threadid, object task, cINT *words, int length, int wentback, int wentpast)

    cdef int process(self, int threadid, cINT *words, int *length, int *wentback, int *wentpast)


