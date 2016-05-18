from pipe.cpipe cimport CPipe
from tools.ctypes cimport *
from numpy cimport *

cdef class DownSample(CPipe):
    cdef:
        uLONG random
        int vocabularysize
        float downsample           # the downsampling parameter
        long totalwords
        cINT* corpusfrequency

        int process(self, int threadid, cINT *words, int *length, int *wentback, int *wentpast)


