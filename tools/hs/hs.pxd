cimport numpy as np
from tools.nnmodel.model cimport model_c
from numpy cimport ndarray

ctypedef np.uint8_t cBYTE
ctypedef np.int32_t cINT
ctypedef np.float32_t cREAL
ctypedef void(*train)(int, cREAL*, int, cREAL*, cREAL*, cINT *, int, cREAL*, float) nogil

cdef class HS:
    cdef void i_stream(self, int id, model_c m, ndarray words, int wentback, int wentpast)
    cdef void setTrainer(self, trainerWrapper t)
    cdef cINT **innernodes
    cdef cBYTE **exp
    cdef int vocsize
    cdef train f

cdef class trainerWrapper:
    cdef train f
    cdef trainerWrapper set(self, train t)

