from arch.SkipgramHScached cimport SkipgramHScached
from tools.types cimport *

cdef class ParagraphVector(SkipgramHScached):
    cdef void learnVector(self, int threadid, int taskid, cINT *words, int length)



