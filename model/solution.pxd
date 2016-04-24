cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.string cimport memset

ctypedef np.int32_t cINT
ctypedef np.uint8_t cBYTE
ctypedef np.uint64_t cULONGLONG
ctypedef np.float32_t cREAL

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cdef inline float min_float(float a, float b) nogil: return a if a <= b else b
cdef inline float max_float(float a, float b) nogil: return a if a >= b else b
cdef inline int min_int(int a, int b) nogil: return a if a <= b else b
cdef inline int max_int(int a, int b) nogil: return a if a >= b else b
cdef inline float abs_float(float a) nogil: return a if a >= 0 else -a

# create a float32 pointer to a numpy array
cdef inline cREAL* toRArray(np.ndarray a):
    return <cREAL *>(np.PyArray_DATA(a))

# create an int32 pointer to a numpy array
cdef inline cINT* toIArray(np.ndarray a):
    return <cINT *>(np.PyArray_DATA(a))

# allocate memory for an array of float32
cdef inline cREAL* allocR(int size) nogil:
    return <cREAL*>malloc(size * sizeof(cREAL))

# allocate memory for an array of float32*
cdef inline cREAL** allocRP(int size) nogil:
    cdef cREAL** r = <cREAL**>malloc(size * sizeof(cREAL*))
    for i in range(size):
        r[i] = NULL
    return r

# allocate memory for an array of int32
cdef inline cINT* allocI(int size) nogil:
    return <cINT*>malloc(size * sizeof(cINT))

cdef inline cINT* allocZeros(int size) nogil:
    cdef cINT *zeros = allocI(size)
    memset(zeros, 0, size * sizeof(cINT))
    return zeros

# allocate memory for an array of int32*
cdef inline cINT** allocIP(int size) nogil:
    cdef cINT** r = <cINT**>malloc(size * sizeof(cINT*))
    for i in range(size):
        r[i] = NULL
    return r

# allocate space for an array of int8
cdef inline cBYTE* allocB(int size) nogil:
    return <cBYTE*>malloc(size * sizeof(cBYTE))

# allocate space for an array of int8*
cdef inline cBYTE** allocBP(int size) nogil:
    cdef cBYTE** r = <cBYTE**>malloc(size * sizeof(cBYTE*))
    for i in range(size):
        r[i] = NULL
    return r

# allocate space for an array of void*
cdef inline void** allocVP(int size) nogil:
    cdef void** r = <void**>malloc(size * sizeof(void*))
    for i in range(size):
        r[i] = NULL
    return r

cdef enum:
    CONTEXTWINDOWSIZE=50000
    TRAINSAMPLESSIZE=1000000

# header for the C representation of model, modelc focusses more on the solution space, and progress
# that is required by
cdef class Solution:
    cdef public object model
    cdef cREAL **layerfw        # array of pointers to layer vectors, created only on demand using getLayer
    cdef cREAL **layerbw        # array of pointers to layer vectors, created only on demand using getLayer
    cdef cREAL **w              # array of pointers to weight matrices
    cdef cINT *w_input          # array that stores the input size of the matrices
    cdef cINT *w_output         # array that stores the output size of the matrices
    cdef public object solution # numpy weight matrices that contain the solution
    cdef public int matrices    # number of matrices

    cdef int MAX_SIGMOID, SIGMOID_TABLE
    cdef cREAL *sigmoidtable    # fast sigmoid lookup table

    cdef cINT *progress         # progress, per thread
    cdef long totalwords        # to estimate the total number of words to be processed,
    cdef int threads            # number of concurrent threads used for learning
    cdef int tasks              # number of dedicated task ids, equals threads unless multiple threads work the same task id
    cdef int split              # 1 assigns every thread a dedicated taskid to process a disjoint subset of words, all theads process on the same data
                                # 0 assigns every data chunk to only one thread
    cdef public float alpha     # initial learning rate

    # for hierarchical softmax
    cdef cINT **innernodes      # array that keep a list of inner nodes that parent each indexed word, the root (id=0) is always last
    cdef cBYTE **exp            # array of the expected values, i.e. whether at a given inner node one must go left=0 or right=1 to find the indexed word
    cdef cINT* word2taskid
    cdef public int singletaskids

    cdef float *getLayerFw(self, int thread, int layer)   # returns a vector to be used as forward layer #layer, unique per thread and layer
    cdef float *getLayerBw(self, int thread, int layer)   # returns a vector to be used as backward layer #layer, unique per thread and layer
    #cdef public int getLayerSize(self, int layer) nogil  # returns the size of the given layer, 0=input, ..., |layers|-1=output
    cdef float *createWorkLayer(self, int layer)          # returns a vector to be used as layer #layere, not shared by he thread
    cdef float *createSigmoidTable(self)                                    # constructs a sigmoid lookup table
    cdef float getProgress(self) nogil                                      # returns a float that indicates the percentage of words processed
    cdef float updateAlpha(self, int threadid, int completed) nogil         # updates the cumber of completed words for the thread and return the new alpha


