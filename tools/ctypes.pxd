cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset

ctypedef np.int32_t cINT
ctypedef np.int64_t cLONG
ctypedef np.uint8_t cBYTE
ctypedef np.uint64_t uLONG
ctypedef np.float32_t cREAL

# cdef extern from "ctypes.h":
#     void* PyCObject_AsVoidPtr(object obj)

cdef inline float min_float(float a, float b) nogil: return a if a <= b else b
cdef inline float max_float(float a, float b) nogil: return a if a >= b else b
cdef inline int min_int(int a, int b) nogil: return a if a <= b else b
cdef inline int max_int(int a, int b) nogil: return a if a >= b else b
cdef inline float abs_float(float a) nogil: return a if a >= 0 else -a

# create a float32 pointer to a numpy array
cdef inline cREAL* toRealArray(np.ndarray a):
    return <cREAL *>(np.PyArray_DATA(a))

# create an int32 pointer to a numpy array
cdef inline cINT* toIntArray(np.ndarray a):
    return <cINT *>(np.PyArray_DATA(a))

# allocate memory for an array of float32
cdef inline cREAL* allocReal(int size) nogil:
    return <cREAL*>malloc(size * sizeof(cREAL))

# allocate memory for an array of float32*
cdef inline cREAL** allocRealP(int size) nogil:
    cdef cREAL** r = <cREAL**>malloc(size * sizeof(cREAL*))
    for i in range(size):
        r[i] = NULL
    return r

# allocate memory for an array of int32
cdef inline cINT* allocInt(int size) nogil:
    return <cINT*>malloc(size * sizeof(cINT))

# allocate memory for an array of int32
cdef inline cINT* allocIntFill(int size, int fill) nogil:
    cdef cINT i,  *r = <cINT*>malloc(size * sizeof(cINT))
    for i in range(size):
        r[i] = fill
    return r

# allocate memory for an array of int32
cdef inline cLONG* allocLong(int size) nogil:
    return <cLONG*>malloc(size * sizeof(cLONG))

# allocate memory for an array of int32
cdef inline uLONG* allocULong(int size) nogil:
    return <uLONG*>malloc(size * sizeof(uLONG))

cdef inline cINT* allocIntZeros(int size) nogil:
    cdef cINT *zeros = allocInt(size)
    memset(zeros, 0, size * sizeof(cINT))
    return zeros

cdef inline uLONG* allocULongZeros(int size) nogil:
    cdef uLONG *zeros = allocULong(size)
    memset(zeros, 0, size * sizeof(uLONG))
    return zeros

# allocate memory for an array of int32*
cdef inline cINT** allocIntP(int size) nogil:
    cdef cINT** r = <cINT**>malloc(size * sizeof(cINT*))
    for i in range(size):
        r[i] = NULL
    return r

# allocate space for an array of int8
cdef inline cBYTE* allocByte(int size) nogil:
    return <cBYTE*>malloc(size * sizeof(cBYTE))

# allocate space for an array of int8
cdef inline cBYTE* allocByteZeros(int size) nogil:
    cdef cBYTE *r = allocByte(size)
    memset(r, 0, size)
    return r

# allocate space for an array of int8*
cdef inline cBYTE** allocByteP(int size) nogil:
    cdef cBYTE** r = <cBYTE**>malloc(size * sizeof(cBYTE*))
    for i in range(size):
        r[i] = NULL
    return r

# allocate space for an array of void*
cdef inline void** allocVoidP(int size) nogil:
    cdef void** r = <void**>malloc(size * sizeof(void*))
    for i in range(size):
        r[i] = NULL
    return r

# allocate space for an array of void
cdef inline void* allocVoid(int length, int elementsize) nogil:
    return <void*>malloc(length * elementsize)



