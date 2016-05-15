from pipe.cpipe cimport CPipe
from model.solution cimport *       # defines the cREAL and cINT types

cdef class contextWindow(CPipe):
    cdef int windowsize
    cdef uLONG random

    cdef void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper,
                     int wlength, int wentback, int wentpast)

