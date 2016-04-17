from __future__ import print_function
from multiprocessing.pool import Pool
from queue import PriorityQueue, Queue, Empty
import warnings
import time
import cython, math
from numpy import float32, int32
from libc.stdio cimport *

from tools.wordio import wordStreamsDecay, wordStreams
from pipe.cy import cypipe

from tools.taketime import taketime
import threading

import numpy as np
cimport numpy as np

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0

# disable output buffering on stdout
setvbuf (stdout, NULL, _IONBF, 0);

# contains the configuration for the network, the solution space (to be used after training) and can build the
# processing pipeline to learn the model.
# For the learning pipeline a c-projection of this model is made to be passed through the pipeline.
class model:
    def __init__(self, alpha,
                 input, vocabulary_input = None, build=[], pipeline=[],
                 mintf=1, cores=2, threads=None, windowsize=5, iterations=1, **kwargs):
        self.__dict__.update(kwargs)
        self.windowsize = windowsize        # windowsize used for w2v models
        self.alpha = alpha                  # initial learning rate
        self.iterations = iterations        # number of times to iterate over the corpus
        self.mintf = mintf                  # minimal collection frequency for terms to be used
        self.build = build                  # functions called with (model) to construct the model
        self.pipeline = pipeline            # python functions that process the input word generator

        # number of cores/threads to use in multithreading mode, by default for every core two
        # threads are used to overcome performance loss by memory blocks
        self.threads = threads if threads is not None else cores * 2

        self.input = input if isinstance(input, list) else self.setupInput(input)
        self.vocabulary_input = vocabulary_input if vocabulary_input is not None else self.setupVocabInput(input)

        self.progress = np.zeros((self.threads), dtype=int32)  # tracks progress per thread
        self.pipe = [[None for x in range(len(self.pipeline))] for y in range(self.threads)]  # first pipelineobject per thread

    # Builds the vocabulay, then build the learning pipeline, and push the inputs through the pipeline.
    def run(self):

        # build the model
        for f in self.build:
            f(self)

        # instantiate the processing pipelines
        self.createPipes()

        # queues for the
        queue = Queue(self.iterations * len(self.input))
        finished = Queue()
        for i in self.input:
            queue.put(i)
        print("running multithreadeded threads %d parts %d iterations %d" %
        ( self.threads, len(self.input), self.iterations))
        if self.threads * len(self.input) == 1 or self.threads == 1:
            learnThread(0, self, queue, finished)
        else:
            threads = []
            for i in range(self.threads):
                t = threading.Thread(target=learnThread, args=(i, self, queue, finished))
                t.daemon = True
                threads.append(t)
                t.start()
        unfinished = self.threads
        starttime = time.time()
        while unfinished > 0:
            try:
                f = finished.get(True, 2)
            except Empty:
                f = None
            if f is not None:
                unfinished -= 1
            p = self.getModelC().getProgressPy();
            if p > 0:
                wps = self.vocab.totalwords * self.iterations * p / (time.time() - starttime)
                alpha = self.modelc.getCurrentAlpha()
                print("progress %4.1f%% wps %d workers %d alpha %f\r" % (100 * p, int(wps), unfinished, alpha), end = '')
            else:
                starttime = time.time()
                wps = 0
        print()

    def setupInput(self, input):
        inputrange = self.inputrange if hasattr(self, 'inputrange') else None
        return wordStreamsDecay(input, parts =self.threads * 4, inputrange=inputrange,
                                windowsize = self.windowsize, iterations = self.iterations)

    def setupVocabInput(self, input):
        inputrange = self.inputrange if hasattr(self, 'inputrange') else None
        return wordStreams(input, parts = self.threads, inputrange=inputrange,
                           windowsize = self.windowsize, iterations = 1)

    def setSolution(self, solution):
        self.solution = solution
        self.getModelC().setSolution(solution)

    def createPipes(self):
        def createPipe(threadid):
            for i in range(len(self.pipeline)):
                self.pipe[threadid][i] = self.pipeline[i](threadid, self)
                if  i > 0:
                    self.pipe[threadid][i].bindTo(self.pipe[threadid][i - 1])

        for t in range(self.threads):
            createPipe(t) # initialize the first to avoid duplicate generation of required shared resources

    # the Cython version of the model, instantiated on first request, which should be after the vocabulary was build.
    def getModelC(self):
        if not hasattr(self, 'modelc'):
            self.modelc = modelc(self)
        return self.modelc

# a thread that repeatedly pulls a chunk of input data from the queue and calls
# the trainer on that chunk, until there is no more input
# note must be a 
def learnThread(threadid, model, queue, finished):
    def learn(input):
        model.pipe[threadid][0].feed(input)

    while True:
        try:
            input = queue.get(False)
            #print("learnThread pull input thread", threadid)
            learn(input)
        except Empty:
            finished.put(threadid)
            break


# C version of the model, to allow nogil Cython modules to process efficiently while accessing the model
# modelc only stores standard model parameters and references to the neural net solution, user defined
# model parameters can be taken from the py-model in the _init_ of a pipe module.
cdef class modelc:
    def __init__(self, m):
        #print("initializing modelc")
        self.model = m                                  # links back to the py-model
        self.progress = allocZeros(m.threads)
        self.totalwords = m.vocab.totalwords * m.iterations  # assumed to be the number of words to be processed (for progress)
        self.windowsize = m.windowsize
        self.vocsize = len(m.vocab)
        self.threads = m.threads
        self.alpha = m.alpha
        self.iterations = m.iterations
        self.sigmoidtable = self.createSigmoidTable()   # used for fast lookup of sigmoid function

    def getModel(self):
        return self.model

    def setSolution(self, solution):
        self.matrices = len(solution)
        self.w = allocRP(self.matrices)
        self.w_input = allocI(self.matrices)
        self.w_output = allocI(self.matrices)
        self.layer = allocRP((self.matrices + 1) * self.threads)
        for l in range(self.matrices):
            self.w[l] = toRArray(solution[l]);
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
    cdef cREAL *getLayer(self, int thread, int layer) nogil:
        cdef int pos = thread * (self.matrices + 1) + layer
        if self.layer[pos] == NULL:
            self.layer[pos] = self.createWorkLayer(layer)
        return self.layer[pos]

    # returns a thread-safe vector for the given layer, 0 being the input and |layer|-1 being the output layer
    # createLayer creates a non-shared layer instance
    cdef cREAL *createWorkLayer(self, int layer) nogil:
        size = self.getLayerSize(layer)
        return allocR(size)

    # returns the size of layer #layer
    cdef cINT getLayerSize(self, int layer) nogil:
        return self.w_input[layer] if layer < self.matrices else self.w_output[layer - 1]

 # returns a float that contains is the fraction of words processed, for reporting and to adjust the learning rate
    cdef float getProgress(self) nogil:
        cdef cREAL currentcompleted = 0
        for i in range(self.threads):
            currentcompleted += self.progress[i]
        return currentcompleted / self.totalwords

    cdef float updateAlpha(self, int threadid, int completed) nogil:
        self.progress[threadid] += completed
        return self.alpha * max_float(1.0 - self.getProgress(), 0.0001)


    def getCurrentAlpha(self):
        return self.alpha * max_float(1.0 - self.getProgress(), 0.0001)

    def getProgressPy(self):
        return self.getProgress()