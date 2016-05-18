from arch.SkipgramNS cimport SkipgramNS
from tools.ctypes cimport *

cdef class SkipgramNScached(SkipgramNS):
    cdef int cachewords, updatecacherate
