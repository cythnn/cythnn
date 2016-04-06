cimport numpy as np
from model.model cimport model_c
from numpy cimport ndarray

ctypedef np.uint8_t cBYTE
ctypedef np.int32_t cINT
ctypedef np.float32_t cREAL

ctypedef void(*followme)(int threadid, model_c, cINT * wordarray, int length, float alpha) nogil

cdef void build_hierarchical_softmax2(model_c, ndarray collectionfrequencies)

cdef void processhs2(int id, model_c m, ndarray words, int wentback, int wentpast)


