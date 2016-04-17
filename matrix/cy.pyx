import cython
import numpy as np
from model.cy cimport *
from numpy import int32, uint64, float32
from tools.taketime import taketime
from numpy cimport ndarray

#creates a list of weight matrices, according to the given LAYER sizes, ordered input, h1, ..., output. The initialization
# functions (always |LAYERS| - 1) are used to seed the weight matrices
#@taketime("createMatrices")
def createMatrices(sizes, init):
    layers = []
    for i in range(len(sizes) - 1):
        if len(init) <= i:
            init.append(None)
        if init[i] is None or init[i] == 0:
            layers.append(np.zeros((sizes[i], sizes[i + 1]), dtype=float32))
        else:
            layers.append(np.empty((sizes[i], sizes[i+1]), dtype=float32))
            randomize(layers[-1])
    return layers

# initializes a weight matrix between layers sized INPUT and OUTPUT with random numbers
cdef void randomize(ndarray array):
    cdef float *a = toRArray(array)
    cdef int i, length = array.shape[0] * array.shape[1]
    cdef int width = array.shape[1]
    cdef unsigned long long random = 1

    with nogil:
        for i in range(length):
            random = random * rand + 11;
            a[i] = ((random & 65535) / 65536.0 - 0.5) / width

# creates a weight matrix between layers sized INPUT and OUTPUT setting all weights to 0
def zeros(input, output):
    return np.zeros((input, output), dtype=float32)
