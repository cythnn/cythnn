from __future__ import print_function
import cython, math
from numpy import float32, int32
from libc.stdio cimport *
import numpy as np
cimport numpy as np
from tools.ctypes cimport *

# The solution is a Cython container that is accessible from Cython modules in the pipeline, allowing nogil Cython modules
# to process efficiently. Typically the solution contains the model's weight matrices w[0], w[1], etc. which is kept in
# shared memory (all threads update the same model), and when requested creates local layer vectors for each thread
# through getLayerFw and getLayerBw to allow thread safe computation of feed forward and back propagation.
# The solution is also a central location to keeps track of progress, and updates the learning parameter alpha.
# The solution arbitrarily also contain some shared values for convenient reuse, such as the sigmoid lookup table.
cdef class Solution:
    def __init__(self, model):
        #print("initializing solution")
        self.progress = allocULongZeros(model.threads + 1)
        self.totalwords = model.vocab.totalwords * model.iterations  # assumed to be the number of words to be processed (for progress)
        self.alpha = model.alpha
        self.threads = model.threads
        self.tasks = model.tasks
        self.SIGMOID_TABLE = 1000
        self.MAX_SIGMOID = 6
        self.sigmoidtable = createSigmoidTable(self.SIGMOID_TABLE, self.MAX_SIGMOID)   # used for fast lookup of sigmoid function

    def setSolution(self, solution):
        self.matrices = len(solution)                   # number of weight matrices in the model
        self.w = allocRealP(self.matrices)                 # references to the weight matrices
        self.w_input = allocInt(self.matrices)            # number of rows in each matrix
        self.w_output = allocInt(self.matrices)           # number of columns in each matrix
        self.layerfw = allocRealP((self.matrices + 1) * self.threads)      # pointers to fw and bw layers, instantiated on request
        self.layerbw = allocRealP((self.matrices + 1) * self.threads)
        for l in range(self.matrices):
            self.w[l] = toRealArray(solution[l]);          # layers are numbered 0,..,n weight matrices 0,..,(n-1)
            self.w_input[l] = solution[l].shape[0]
            self.w_output[l] = solution[l].shape[1]

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
        return allocReal(size)

    # returns the size of layer #layer
    def getLayerSize(self, layer):
        return self.w_input[layer] if layer < self.matrices else self.w_output[layer - 1]

    def getTotalWords(self):
        return self.totalwords

    def setTotalWords(self, totalwords):
        self.totalwords = totalwords

    # returns a float that contains is the fraction of words processed, for reporting and to adjust the learning rate
    cdef float getProgress(self) nogil:
        cdef int i
        cdef float currentcompleted = 0
        for i in range(self.threads + 1):
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

