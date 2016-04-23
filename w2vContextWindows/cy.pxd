from model.cpipe cimport CPipe
from model.solution cimport *       # defines the cREAL and cINT types

cdef class contextWindow(CPipe):
    cdef int windowsize
    cdef cULONGLONG random
    cdef int vocabularysize
    cdef float sample
    cdef long totalwords
    cdef cINT* corpusfrequency
    cdef int debug

    cdef int process(self, int threadid, cINT *words, cINT *clower, cINT *cupper,
                     int wlength, int wentback, int wentpast)

