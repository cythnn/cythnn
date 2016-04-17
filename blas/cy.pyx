import cython
import scipy.linalg.blas as fblas
import numpy as np

REAL = np.float32
INT = np.int32

# from gensim
# bind scipy blas functions

# scopy: y = x
cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)

# saxpy: y = y + a * x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)

# sdot: (single precision float) = x * y
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)

# dsdot: (double precision) = x * y
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)

# snrm2: = sqrt( x * x )
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)

# sscal: x = a * x
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer)