cimport numpy as np

#
# External imports from Basic Linear Algebra Subroutines (BLAS)
#

cdef extern from "Python.h":

    cdef void Py_INCREF(object)


cdef extern from "numpy/arrayobject.h":

    cdef void import_array()

    cdef object PyArray_ZEROS(int nd, np.npy_intp *dims, int typenum, int fortran)
    cdef object PyArray_SimpleNew(int nd, np.npy_intp *dims, int typenum)
    cdef object PyArray_EMPTY(int nd, np.npy_intp *dims, int typenum, int fortran)

    int PyArray_ISCARRAY(np.ndarray instance) # I can't get this one to work?!?

    int NPY_FLOAT   # PyArray_FLOAT  deprecated.
    int NPY_DOUBLE  # PyArray_DOUBLE deprecated.


cdef extern from "cblas.h":

    enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
    enum CBLAS_UPLO:      CblasUpper, CblasLower
    enum CBLAS_DIAG:      CblasNonUnit, CblasUnit
    enum CBLAS_SIDE:      CblasLeft, CblasRight

    ###########################################################################
    # BLAS level 1 routines
    ###########################################################################

    # Swap vectors: x <-> y
    void   lib_sswap  "cblas_sswap"(int M, float  *x, int dx, float  *y, int dy) nogil
    void   lib_dswap  "cblas_dswap"(int M, double *x, int dx, double *y, int dy) nogil

    # Scale a vector: x <- alpha*x
    void   lib_sscal  "cblas_sscal"(int N, float  alpha, float  *x, int dx) nogil
    void   lib_dscal  "cblas_dscal"(int N, double alpha, double *x, int dx) nogil

    # Copy a vector: y <- x
    void   lib_scopy  "cblas_scopy"(int N, float  *x, int dx, float  *y, int dy) nogil
    void   lib_dcopy  "cblas_dcopy"(int N, double *x, int dx, double *y, int dy) nogil

    # Combined multiply/add: y <- alpha*x + y
    void   lib_saxpy  "cblas_saxpy"(int N, float  alpha, float  *x, int dx,
                                                         float  *y, int dy) nogil
    void   lib_daxpy  "cblas_daxpy"(int N, double alpha, double *x, int dx,
                                                         double *y, int dy) nogil

    # Dot product: x'y
    float  lib_sdot   "cblas_sdot"(int N, float  *x, int dx, float  *y, int dy) nogil
    double lib_ddot   "cblas_ddot"(int N, double *x, int dx, double *y, int dy) nogil

    # Euclidian (2-)norm: ||x||_2
    float  lib_snrm2  "cblas_snrm2"(int N, float  *x, int dx) nogil
    double lib_dnrm2  "cblas_dnrm2"(int N, double *x, int dx) nogil

    # One norm: ||x||_1 = sum |xi|
    float  lib_sasum  "cblas_sasum"(int N, float  *x, int dx) nogil
    double lib_dasum  "cblas_dasum"(int N, double *x, int dx) nogil

    # Argmax: i = arg max(|xj|)
    int    lib_isamax "cblas_isamax"(int N, float  *x, int dx) nogil
    int    lib_idamax "cblas_idamax"(int N, double *x, int dx) nogil

    # Generate a plane rotation.
    void   lib_srotg  "cblas_srotg"(float  *a, float  *b, float  *c, float  *s) nogil
    void   lib_drotg  "cblas_drotg"(double *a, double *b, double *c, double *s) nogil

    # Generate a modified plane rotation.
    void   lib_srotmg "cblas_srotmg"(float  *d1, float  *d2, float  *b1,
                                     float  b2, float  *P) nogil
    void   lib_drotmg "cblas_drotmg"(double *d1, double *d2, double *b1,
                                     double b2, double *P) nogil

    # Apply a plane rotation.
    void   lib_srot   "cblas_srot"(int N, float  *x, int dx,
                                          float  *y, int dy,
                                          float c, float s) nogil
    void   lib_drot   "cblas_drot"(int N, double *x, int dx,
                                          double *y, int dy,
                                          double c, double s) nogil

    # Apply a modified plane rotation.
    void   lib_srotm  "cblas_srotm"(int N, float *x, int dx,
                                           float *y, int dy,
                                           float *P) nogil
    void   lib_drotm  "cblas_drotm"(int N, double *x, int dx,
                                           double *y, int dy,
                                           double *P) nogil

    ###########################################################################
    # BLAS level 2 routines
    ###########################################################################

    # Combined multiply/add: y <- alpha*Ax + beta*y or y <- alpha*A'x + beta*y
    void lib_sgemv "cblas_sgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 int M, int N, float  alpha, float  *A, int lda,
                                               float  *x, int dx,
                                 float  beta,  float  *y, int dy) nogil

    void lib_dgemv "cblas_dgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 int M, int N, double alpha, double *A, int lda,
                                               double *x, int dx,
                                 double beta,  double *y, int dy) nogil

    # Rank-1 update: A <- alpha * x*y' + A
    void lib_sger  "cblas_sger"(CBLAS_ORDER Order, int M, int N, float  alpha,
                                float  *x, int dx, float  *y, int dy,
                                float  *A, int lda) nogil

    void lib_dger  "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                                double *x, int dx, double *y, int dy,
                                double *A, int lda) nogil

    ###########################################################################
    # BLAS level 3 routines
    ###########################################################################

    void lib_sgemm "cblas_sgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 float  alpha, float  *A, int lda, float  *B, int ldb,
                                 float  beta, float  *C, int ldc) nogil

    void lib_dgemm "cblas_dgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 double alpha, double *A, int lda, double *B, int ldb,
                                 double beta, double *C, int ldc) nogil


#####################################
#
# BLAS LEVEL 1 (vector operations)
#
#####################################

# vector swap: x <-> y
cdef void sswap_(int M, float *x, int dx, float *y, int dy) nogil

cdef void dswap_(int M, double *x, int dx, double *y, int dy) nogil

# scalar vector multiply: x *= alpha
cdef void sscal_(int N, float alpha, float *x, int dx) nogil

cdef void dscal_(int N, double alpha, double *x, int dx) nogil

# vector copy: y <- x
cdef void scopy_(int N, float *x, int dx, float *y, int dy) nogil

cdef void dcopy_(int N, double *x, int dx, double *y, int dy) nogil

# vector addition: y += alpha*x
cdef void saxpy_(int N, float alpha, float *x, int dx, float *y, int dy) nogil

cdef void daxpy_(int N, double alpha, double *x, int dx, double *y, int dy) nogil

# vector dot product: x.T y
cdef float sdot_(int N, float *x, int dx, float *y, int dy) nogil

cdef double ddot_(int N, double *x, int dx, double *y, int dy) nogil

# Euclidean norm:  ||x||_2
cdef float snrm2_(int N, float *x, int dx) nogil

cdef double dnrm2_(int N, double *x, int dx) nogil

# sum of absolute values: ||x||_1
cdef float sasum_(int N, float *x, int dx) nogil

cdef double dasum_(int N, double *x, int dx) nogil

# index of maximum absolute value element
cdef int isamax_(int N, float *x, int dx) nogil

cdef int idamax_(int N, double *x, int dx) nogil

###########################################
#
# BLAS LEVEL 2 (matrix-vector operations)
#
###########################################


# single precision general matrix-vector multiply
cdef void sgemv_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                    float  alpha, float  *A, int lda, float  *x, int dx,
                    float  beta, float  *y, int dy) nogil

# double precision general matrix-vector multiply
cdef void dgemv_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                    double alpha, double *A, int lda, double *x, int dx,
                    double beta, double *y, int dy) nogil

####

# single precision rank-1 opertion (aka outer product)
cdef void sger_(CBLAS_ORDER Order, int M, int N, float  alpha, float  *x, int dx,
                    float  *y, int dy, float  *A, int lda) nogil

# double precision rank-1 opertion (aka outer product)
cdef void dger_(CBLAS_ORDER Order, int M, int N, double alpha, double *x, int dx,
                    double *y, int dy, double *A, int lda) nogil

####################################################
#
# BLAS LEVEL 3 (matrix-matrix operations)
#
####################################################


# single precision general matrix-matrix multiply
cdef void sgemm_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, float alpha, float *A, int lda, float *B,
                 int ldb, float beta, float *C, int ldc) nogil

# double precision general matrix-matrix multiply
cdef void dgemm_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, double alpha, double *A, int lda, double *B,
                 int ldb, double beta, double *C, int ldc) nogil



