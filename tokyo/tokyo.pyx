cimport numpy as np
import ctypes
from ctypes.util import find_library

import_array()

def setMaxThreads(int threads):
    try_paths = ['/opt/OpenBLAS/lib/libopenblas.so',
                 '/lib/libopenblas.so',
                 '/usr/lib/libopenblas.so.0',
                 '/usr/lib/openblas/libopenblas.so.0',
                 find_library('openblas')]
    openblas_lib = None
    for libpath in try_paths:
        try:
            openblas_lib = ctypes.cdll.LoadLibrary(libpath)
            break
        except OSError:
            continue
    if openblas_lib is None:
        raise EnvironmentError('Could not locate an OpenBLAS shared library', 2)
    openblas_lib.openblas_set_num_threads(threads)

##########################################################################
# BLAS LEVEL 1
##########################################################################

# Each subroutine comes in two variants:
# [sd]name and [sd]name_
# The variant with the trailing underscore skips type and dimension checks,
# calls the low-level C-routine directly and works with C types.

# vector swap: x <-> y
cdef void sswap_(int M, float *x, int dx, float *y, int dy) nogil:
    lib_sswap(M, x, dx, y, dy)

cdef void dswap_(int M, double *x, int dx, double *y, int dy) nogil:
    lib_dswap(M, x, dx, y, dy)

# scalar vector multiply: x *= alpha
cdef void sscal_(int N, float alpha, float *x, int dx) nogil:
    lib_sscal(N, alpha, x, dx)

cdef void dscal_(int N, double alpha, double *x, int dx) nogil:
    lib_dscal(N, alpha, x, dx)

# vector copy: y <- x
cdef void scopy_(int N, float *x, int dx, float *y, int dy) nogil:
    lib_scopy(N, x, dx, y, dy)

cdef void dcopy_(int N, double *x, int dx, double *y, int dy) nogil:
    lib_dcopy(N, x, dx, y, dy)

# vector addition: y += alpha*x
cdef void saxpy_(int N, float alpha, float *x, int dx, float *y, int dy) nogil:
    lib_saxpy(N, alpha, x, dx, y, dy)

cdef void daxpy_(int N, double alpha, double *x, int dx, double *y, int dy) nogil:
    lib_daxpy(N, alpha, x, dx, y, dy)

# vector dot product: x'y
cdef float sdot_(int N, float *x, int dx, float *y, int dy) nogil:
    return lib_sdot(N, x, dx, y, dy)

cdef double ddot_(int N, double *x, int dx, double *y, int dy) nogil:
    return lib_ddot(N, x, dx, y, dy)

# Euclidean norm:  ||x||_2
cdef float snrm2_(int N, float *x, int dx) nogil:
    return lib_snrm2(N, x, dx)

cdef double dnrm2_(int N, double *x, int dx) nogil:
    return lib_dnrm2(N, x, dx)

# sum of absolute values: ||x||_1
cdef float sasum_(int N, float *x, int dx) nogil:
    return lib_sasum(N, x, dx)

cdef double dasum_(int N, double *x, int dx) nogil:
    return lib_dasum(N, x, dx)

# index of maximum absolute value element
cdef int isamax_(int N, float *x, int dx) nogil:
    return lib_isamax(N, x, dx)

cdef int idamax_(int N, double *x, int dx) nogil:
    return lib_idamax(N, x, dx)

# Generate a modified Givens plane rotation.
cdef void srotmg_(float *d1, float *d2, float *x, float y, float *param) nogil:
    lib_srotmg(d1, d2, x, y, param)

cdef void drotmg_(double *d1, double *d2, double *x, double y, double *param) nogil:
    lib_drotmg(d1, d2, x, y, param)

# Apply a Givens plane rotation.
cdef void srot_(int N, float *x, int dx, float *y, int dy, float c, float s) nogil:
    lib_srot(N, x, dx, y, dy, c, s)

cdef void drot_(int N, double *x, int dx, double *y, int dy, double c, double s) nogil:
    lib_drot(N, x, dx, y, dy, c, s)

# Apply a modified Givens plane rotation.
cdef void srotm_(int N, float *x, int dx, float *y, int dy, float *param) nogil:
    lib_srotm(N, x, dx, y, dy, param)

cdef void drotm_(int N, double *x, int dx, double *y, int dy, double *param) nogil:
    lib_drotm(N, x, dx, y, dy, param)

##########################################################################
# BLAS LEVEL 2
##########################################################################

#
# matrix times vector: A = alpha * A   x + beta * y
#                  or  A = alpha * A.T x + beta * y
#
# single precison

cdef void sgemv_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                    float alpha, float *A, int lda, float *x, int dx,
                    float beta, float *y, int dy) nogil:
    lib_sgemv(Order, TransA, M, N, alpha, A, lda, x, dx, beta, y, dy)

# double precision

cdef void dgemv_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                    double alpha, double *A, int lda, double *x, int dx,
                    double beta, double *y, int dy) nogil:
    lib_dgemv(Order, TransA, M, N, alpha, A, lda, x, dx, beta, y, dy)

#
# vector outer-product: A = alpha * outer_product(x, y.T)
#

# Note: when calling this make sure you're working with a buffer otherwise
# a whole lot of Python stuff will be going before the call to this function
# is made in order to get the size of the arrays, there the data is located...

# single precision

cdef void sger_(CBLAS_ORDER Order, int M, int N, float alpha, float *x, int dx,
                float *y, int dy, float *A, int lda) nogil:

    lib_sger(Order, M, N, alpha, x, dx, y, dy, A, lda)

# double precision

cdef void dger_(CBLAS_ORDER Order, int M, int N, double alpha, double *x, int dx,
                double *y, int dy, double *A, int lda) nogil:

    lib_dger(Order, M, N, alpha, x, dx, y, dy, A, lda)

##########################################################################
#
# BLAS LEVEL 3
#
##########################################################################


# matrix times matrix: C = alpha * A   B   + beta * C
#                  or  C = alpha * A.T B   + beta * C
#                  or  C = alpha * A   B.T + beta * C
#                  or  C = alpha * A.T B.T + beta * C
#
# single precision

cdef void sgemm_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, float alpha, float *A, int lda, float *B,
                 int ldb, float beta, float *C, int ldc) nogil:

    lib_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)

# matrix times matrix: C = alpha * A   B   + beta * C
#                  or  C = alpha * A.T B   + beta * C
#                  or  C = alpha * A   B.T + beta * C
#                  or  C = alpha * A.T B.T + beta * C
#
# double precision

cdef void dgemm_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, double alpha, double *A, int lda, double *B,
                 int ldb, double beta, double *C, int ldc) nogil:

    lib_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)




