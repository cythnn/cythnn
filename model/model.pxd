cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

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

cdef inline float min_float(float a, float b) nogil: return a if a <= b else b
cdef inline float max_float(float a, float b) nogil: return a if a >= b else b
cdef inline float abs_float(float a) nogil: return a if a >= 0 else -a

cdef inline cREAL* toRArray(np.ndarray a):
    return <cREAL *>(np.PyArray_DATA(a))

cdef inline cINT* toIArray(np.ndarray a):
    return <cINT *>(np.PyArray_DATA(a))

cdef inline cREAL* allocR(int size) nogil:
    return <cREAL*>malloc(size * sizeof(cREAL))

cdef inline cREAL** allocRP(int size) nogil:
    cdef cREAL** r = <cREAL**>malloc(size * sizeof(cREAL*))
    for i in range(size):
        r[i] = NULL
    return r

cdef inline cINT* allocI(int size) nogil:
    return <cINT*>malloc(size * sizeof(cINT))

cdef inline cINT** allocIP(int size) nogil:
    cdef cINT** r = <cINT**>malloc(size * sizeof(cINT*))
    for i in range(size):
        r[i] = NULL
    return r

cdef inline cBYTE* allocB(int size) nogil:
    return <cBYTE*>malloc(size * sizeof(cBYTE))

cdef inline cBYTE** allocBP(int size) nogil:
    cdef cBYTE** r = <cBYTE**>malloc(size * sizeof(cBYTE*))
    for i in range(size):
        r[i] = NULL
    return r

cdef inline void** allocVP(int size) nogil:
    cdef void** r = <void**>malloc(size * sizeof(void*))
    for i in range(size):
        r[i] = NULL
    return r

cdef class model_c:
    cdef cREAL **layer, **w, *sigmoidtable
    cdef int windowsize, vectorsize, totalwords, cores, vocsize
    cdef int iterations, matrices, debugtarget, MAX_SIGMOID, SIGMOID_TABLE
    cdef cINT *w_input, *w_output, *progress
    cdef float alpha

    # for hierarchical softmax
    cdef cINT **innernodes
    cdef cBYTE **exp

    cdef void **pipelinec

    cdef cREAL * getLayer(self, int thread, int layer) nogil
    cdef cINT getLayerSize(self, int layer) nogil
    cdef cREAL* createSigmoidTable(self)
    cdef float getProgress(self) nogil
    cdef void addPipeline(self, void *f)
    cdef void* getPipeline(self, int index)

