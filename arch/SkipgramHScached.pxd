from arch.SkipgramHS cimport SkipgramHS
from tools.ctypes cimport *

cdef class SkipgramHScached(SkipgramHS):
    cdef int cachewords, cacheinner, updatecacherate



