import scipy.linalg.blas as fblas
cimport numpy as np

ctypedef np.float32_t cREAL
ctypedef np.int32_t cINT

cdef extern from "ctypes.h":
    void* PyCObject_AsVoidPtr(object obj)

ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil
ctypedef void (*sswap_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil

cdef:
    # scopy: y = x
    scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)

    # saxpy: y = y + a * x
    saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)

    # sdot: (single precision float) = x * y
    sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)

    # snrm2: = sqrt( x * x )
    snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)

    # sscal: x = a * x
    sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer)

    # sswap: x,y = y,x
    sswap_ptr sswap=<sswap_ptr>PyCObject_AsVoidPtr(fblas.sswap._cpointer)
