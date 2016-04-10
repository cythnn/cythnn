from pipe.cy cimport cypipe
from model.cy cimport *
from numpy cimport ndarray

# the successor receives a position for the target word, with clower and cupper that mark
# the window context, in the array of ids of words
ctypedef void(*outputMethod)(self, cINT * words, cINT *clower, cINT *cupper, int length) nogil

cdef class contextWindow(cypipe):
    cdef outputMethod successorMethod
    cdef int windowsize
    cdef cULONGLONG random

    cdef void feed2process(self, ndarray wordids, int wentback, int wentpast)

    cdef void process(self, cINT *words, int wlength, int wentback, int wentpast) nogil

