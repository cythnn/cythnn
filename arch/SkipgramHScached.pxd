from arch.SkipgramHS cimport SkipgramHS
from tools.types cimport *

cdef class SkipgramHScached(SkipgramHS):
    cdef int cachewords, cacheinner, updatecacherate



