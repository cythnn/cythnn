import cython
from numpy cimport ndarray

cdef void randomize(ndarray array)

cdef void randomize1(ndarray array)

cdef void randomize2(ndarray array, int width)
