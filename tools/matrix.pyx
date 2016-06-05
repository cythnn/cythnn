import numpy as np
from numpy import int32, uint64, float32
from numpy cimport ndarray
from libc.stdio cimport *
from model.solution cimport *
from blas cimport sdot, sswap, snrm2, sscal

cdef uLONG rand_prime = uint64(25214903917)
cdef int iONE = 1

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
    cdef float *a = toRealArray(array)
    cdef int i, length = array.shape[0] * array.shape[1]
    cdef int width = array.shape[1]
    cdef unsigned long long random = 1

    with nogil:
        for i in range(length):
            random = random * rand + 11;
            a[i] = ((random & 65535) / 65536.0 - 0.5) / width

# initializes a weight matrix between layers sized INPUT and OUTPUT with random numbers
cdef void randomize1(ndarray array):
    cdef float *a = toRealArray(array)
    cdef int i, length = len(array)
    cdef unsigned long long random = 1

    with nogil:
        for i in range(length):
            random = random * rand_prime + 11;
            a[i] = ((random & 65535) / 65536.0 - 0.5)

cdef void randomize2(ndarray array, int width):
    cdef float s, min, *a = toRealArray(array)
    cdef int minj, i, j, length = len(array)

    randomize1(array)
    with nogil:
        printf("", length)
        printf("", i)
        i = 0
        while i < length - 2 * width:
            min = sdot(&width, &a[i], &iONE, &a[i + width], &iONE)
            minj = i + width
            j = minj + width
            while j < length:
                s = sdot(&width, &a[i], &iONE, &a[j], &iONE)
                if minj == -1 or s < min:
                    min = s
                    minj = j
                j += width
            if minj > i + width:
                sswap(&width, &a[i + width], &iONE, &a[minj], &iONE)
            i += width

# creates a weight matrix between layers sized INPUT and OUTPUT setting all weights to 0
def zeros(input, output):
    return np.zeros((input, output), dtype=float32)

def normalize(nparray):
    cdef:
        int length = nparray.shape[0]
        int vectorsize = nparray.shape[1]
        cREAL *array = toRealArray(nparray)
        int i
        float magnitude

    print("normalize", length, vectorsize)
    with nogil:
        for i in range(length):
            magnitude = 1 / snrm2(&vectorsize, &array[i * vectorsize], &iONE)
            sscal( &vectorsize, &magnitude, &array[i * vectorsize], &iONE)
