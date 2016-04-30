from model.solution cimport Solution

cdef class CPipe:
    cdef public object learner
    cdef public object model
    cdef Solution solution
    cdef public int pipeid
