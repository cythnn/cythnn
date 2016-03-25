import cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc
from numpy cimport int32_t
import scipy.linalg.blas as fblas

REAL = np.float32
cdef int ONE = 1
cdef int ZERO = 0

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
#cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

class NN:
    tables = []

    def __init__(self):
        print("aap")

    def addZeros(self, rows, columns):
        self.tables.append(np.zeros((rows, columns), dtype=REAL))

    def addOnes(self, rows, columns):
        self.tables.append(np.ones((rows, columns), dtype=REAL))

    def addEmpty(self, rows, columns):
        self.tables.append(np.empty((rows, columns), dtype=REAL))

    def add(self, arr):
        self.tables.append(arr.astype(dtype=REAL, subok=False, copy=False))

    def finalize(self):
        if not hasattr(self, 'nn'):
            self.nn = nn(np.array(self.tables))
        return self.nn

    def sdot(self, t1, r1, t2, r2):
        return self.nn.sdot(t1,r1,t2,r2)

    def scopy(self, t1, r1, t2, r2):
        return self.nn.scopy(t1,r1,t2,r2)

    def saxpy(self, a, t1, r1, t2, r2):
        return self.nn.saxpy(a, t1,r1,t2,r2)

    def sscal(self, a, t1, r1):
        return self.nn.sscal(a, t1,r1)

    def snrm2(self, t1, r1):
        return self.nn.snrm2(t1,r1)

cdef REAL_t* toCArray(np.ndarray a):
    return <REAL_t *>(np.PyArray_DATA(a))

cdef class nn:
    def __init__(self, np.ndarray [:] arr):
        self.rows = np.asarray([ t.shape[0] for t in arr ], dtype=np.int32)
        self.columns = np.asarray([ t.shape[1] for t in arr ], dtype=np.int32)
        self.table = <REAL_t**>malloc(len(arr) * sizeof(REAL_t**))
        for i in range(len(arr)):
            self.table[i] = toCArray(arr[i])

    def sdot(self, int t1, int r1, int t2, int r2):
        r1 *= self.columns[t1]
        r2 *= self.columns[t2]
        return sdot(&(self.columns[t1]), &(self.table[t1][r1]), &ONE, &(self.table[t2][r2]), &ONE)

    def sscal(self, int t1, int r1, float alpha):
        r1 *= self.columns[t1]
        return sscal(&(self.columns[t1]), &alpha, &(self.table[t1][r1]), &ONE)

    def snrm2(self, int t1, int r1):
        r1 *= self.columns[t1]
        return snrm2(&(self.columns[t1]), &(self.table[t1][r1]), &ONE)

    def scopy(self, int t1, int r1, int t2, int r2):
        r1 *= self.columns[t1]
        r2 *= self.columns[t2]
        return scopy(&(self.columns[t1]), &(self.table[t1][r1]), &ONE, &(self.table[t2][r2]), &ONE)

    def saxpy(self, float alpha, int t1, int r1, int t2, int r2):
        r1 *= self.columns[t1]
        r2 *= self.columns[t2]
        return saxpy(&(self.columns[t1]), &alpha, &(self.table[t1][r1]), &ONE, &(self.table[t2][r2]), &ONE)

    
