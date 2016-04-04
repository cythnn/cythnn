cimport numpy as np
from libc.stdlib cimport malloc

ctypedef np.int32_t cINT
ctypedef np.uint8_t cBYTE
ctypedef np.uint64_t cULONGLONG
ctypedef np.float32_t cREAL
ctypedef int (*inputFunction)(model_c, void*)
ctypedef void* (*initFunction)(model_c)

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cdef inline cREAL* toRArray(np.ndarray a):
    return <cREAL *>(np.PyArray_DATA(a))

cdef inline cINT* toIArray(np.ndarray a):
    return <cINT *>(np.PyArray_DATA(a))

cdef inline cREAL* allocR(int size):
    return <cREAL*>malloc(size * sizeof(cREAL))

cdef inline cREAL** allocRP(int size):
    cdef cREAL** r = <cREAL**>malloc(size * sizeof(cREAL*))
    for i in range(size):
        r[i] = NULL
    return r

cdef inline cINT* allocI(int size):
    return <cINT*>malloc(size * sizeof(cINT))

cdef inline cINT** allocIP(int size):
    cdef cINT** r = <cINT**>malloc(size * sizeof(cINT*))
    for i in range(size):
        r[i] = NULL
    return r

cdef inline cBYTE* allocB(int size):
    return <cBYTE*>malloc(size * sizeof(cBYTE))

cdef inline cBYTE** allocIB(int size):
    cdef cBYTE** r = <cBYTE**>malloc(size * sizeof(cBYTE*))
    for i in range(size):
        r[i] = NULL
    return r

cdef class model_c:
    cdef cREAL **layer, **w, *exptable
    cdef int windowsize, vectorsize, totalwords, cores, maxdepth, vocsize, iterations, matrices, threads, target
    cdef cINT *inner, *exp, *w_input, *w_output
    cdef float alpha
    cdef void *trainer

    cdef cREAL * getLayer(self, int thread, int layer)