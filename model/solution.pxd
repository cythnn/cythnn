cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.string cimport memset
from tools.types cimport *

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0

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

    cdef cLONG *progress         # progress, per thread
    cdef long totalwords        # to estimate the total number of words to be processed,
    cdef int threads            # number of concurrent threads used for learning
    cdef int tasks              # number of dedicated task ids, equals threads unless multiple threads work the same task id
    cdef public float alpha     # initial learning rate

    # for hierarchical softmax
    cdef cINT **innernodes      # array that keep a list of inner nodes that parent each indexed word, the root (id=0) is always last
    cdef cBYTE **exp            # array of the expected values, i.e. whether at a given inner node one must go left=0 or right=1 to find the indexed word

    cdef float *getLayerFw(self, int thread, int layer)   # returns a vector to be used as forward layer #layer, unique per thread and layer
    cdef float *getLayerBw(self, int thread, int layer)   # returns a vector to be used as backward layer #layer, unique per thread and layer
    #cdef public int getLayerSize(self, int layer) nogil  # returns the size of the given layer, 0=input, ..., |layers|-1=output
    cdef float *createWorkLayer(self, int layer)          # returns a vector to be used as layer #layere, not shared by he thread
    cdef float *createSigmoidTable(self)                                    # constructs a sigmoid lookup table
    cdef float getProgress(self) nogil                                      # returns a float that indicates the percentage of words processed
    cdef float updateAlpha(self, int threadid, int completed) nogil         # updates the cumber of completed words for the thread and return the new alpha


