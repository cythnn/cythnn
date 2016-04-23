from __future__ import print_function
import cython, math
from numpy import float32, int32
from libc.stdio cimport *
import numpy as np
cimport numpy as np

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0

# The solution is a Cython container that is accessible from Cython modules in the pipeline, allowing nogil Cython modules
# to process efficiently. Typically the solution contains the model's weight matrices w[0], w[1], etc., can provide temporary
# thread save layer vectors through getLayerFw and getLayerBw to allow separate computation of feed forward and back propagation.
# Finally, the solution keeps track of progress, and updates the learning parameter alpha.
# The solution arbitrarily also contain some shared values for convenient reuse, such as the sigmoid lookup table.
cdef class Solution:
    def __init__(self, model):
        #print("initializing solution")
        self.progress = allocZeros(model.threads)
        self.totalwords = model.vocab.totalwords * model.iterations  # assumed to be the number of words to be processed (for progress)
        self.alpha = model.alpha
        self.threads = model.threads
        self.sigmoidtable = self.createSigmoidTable()   # used for fast lookup of sigmoid function

    def setSolution(self, solution):
        self.matrices = len(solution)                   # number of weight matrices in the model
        self.w = allocRP(self.matrices)                 # references to the weight matrices
        self.w_input = allocI(self.matrices)            # number of rows in each matrix
        self.w_output = allocI(self.matrices)           # number of columns in each matrix
        self.layerfw = allocRP((self.matrices + 1) * self.threads)      # pointers to fw and bw layers, instantiated on request
        self.layerbw = allocRP((self.matrices + 1) * self.threads)
        for l in range(self.matrices):
            self.w[l] = toRArray(solution[l]);          # layers are numbered 0,..,n weight matrices 0,..,(n-1)
            self.w_input[l] = solution[l].shape[0]
            self.w_output[l] = solution[l].shape[1]

    # fast lookup table for sigmoid activation function
    cdef cREAL* createSigmoidTable(self):
        self.MAX_SIGMOID = 6
        self.SIGMOID_TABLE = 1000
        cdef cREAL* table = allocR(self.SIGMOID_TABLE)
        for i in range(self.SIGMOID_TABLE):
            e = math.exp(float32(2 * self.MAX_SIGMOID * i / self.SIGMOID_TABLE - self.MAX_SIGMOID))
            table[i] = e / float32(e + 1)
        return table

    # returns a thread-safe vector for the given layer, 0 being the input and |layer|-1 being the output layer
    # getLayer allows the layer to be shared over different pipe modules in the same thread
    cdef cREAL *getLayerFw(self, int thread, int layer):
        cdef int pos = thread * (self.matrices + 1) + layer
        if self.layerfw[pos] == NULL:
            self.layerfw[pos] = self.createWorkLayer(layer)
        return self.layerfw[pos]

    # returns a thread-safe vector for the given layer, 0 being the input and |layer|-1 being the output layer
    # getLayer allows the layer to be shared over different pipe modules in the same thread
    cdef cREAL *getLayerBw(self, int thread, int layer):
        cdef int pos = thread * (self.matrices + 1) + layer
        if self.layerbw[pos] == NULL:
            self.layerbw[pos] = self.createWorkLayer(layer)
        return self.layerbw[pos]

    # returns a thread-safe vector for the given layer, 0 being the input and |layer|-1 being the output layer
    # createLayer creates a non-shared layer instance
    cdef cREAL *createWorkLayer(self, int layer):
        size = self.getLayerSize(layer)
        return allocR(size)

    # returns the size of layer #layer
    def getLayerSize(self, layer):
        return self.w_input[layer] if layer < self.matrices else self.w_output[layer - 1]

    # returns a float that contains is the fraction of words processed, for reporting and to adjust the learning rate
    cdef float getProgress(self) nogil:
        cdef int i
        cdef float currentcompleted = 0
        for i in range(self.threads):
            currentcompleted += self.progress[i]
        return currentcompleted / self.totalwords

    # updates the number of processed words, and returns the updated alpha
    cdef float updateAlpha(self, int threadid, int completed) nogil:
        self.progress[threadid] += completed
        return self.alpha * max_float(1.0 - self.getProgress(), 0.0001)

    # updates completed words outside tasks, for instance words skipped by a preprocessor
    def updateProcessed(self, completed):
        self.progress[self.threads] += completed

    # returns the current learning rate, used for reporting
    def getCurrentAlpha(self):
        return self.alpha * max_float(1.0 - self.getProgress(), 0.0001)

    # returns the (estimated) processed percentage
    def getProgressPy(self):
        return self.getProgress()
