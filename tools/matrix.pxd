import cython
from numpy cimport ndarray

cdef:
    void randomize(ndarray array)

    void randomize1(ndarray array)

    void randomize2(ndarray array, int width)
