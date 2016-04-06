from queue import Queue, Empty

import cython, math
from numpy import float32, int32
from blas.blas cimport sdot, saxpy
import threading

import numpy as np
cimport numpy as np

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0

# contains the configuration for the network, the solution space (to be used after training) and can build the
# processing pipeline to learn the model.
# For the learning pipeline a c-projection of this model is made to be passed through the pipeline.
class model:
    def __init__(self, alpha,
                 input, build, pipeline, pipelinec,
                 mintf=1, cores=1, windowsize=5, iterations=1, debugtarget=None, **kwargs):
        self.__dict__.update(kwargs)
        self.windowsize = windowsize        # windowsize used for w2v models
        self.start_alpha = alpha            # initial learning rate
        self.iterations = iterations        # number of times to iterate over the corpus
        self.mintf = mintf                  # minimal collection frequency for terms to be used
        self.build = build                  # functions called with (model) to construct the model
        self.pipeline = pipeline            # python functions that process the input word generator
        self.pipelinec = pipelinec          # cython functions that complete the pipeline
        self.cores = cores                  # number of cores to use in multithreading mode
        self.input = input                  # reusable list of generators that generate input, used twice to build the
                                            # vocabulary and reused to learn the model
        self.progress = np.zeros((cores), dtype=int32)  # tracks progress per core
        self.debugtarget = debugtarget  # used for debugging purposes

    # Builds the vocabulay, then build the learning pipeline, and push the inputs through the pipeline.
    def run(self):
        for f in self.build:
            f(self)
        for f in self.pipelinec:
            f(self)

        #self.solution = self.buildsolution(model = self)

        #self.getModelC().generatePipeline(self.pipelinec)

        # when there is only one core or one input, run in single mode,
        # otherwise run multithreaded, learning the model in parallel.
        # The current version requires the cores to be on a single machine with shared memory.
        if len(self.input) == 1 or self.cores == 1:
            for w in self.input:
                model.learn(threadid=0, feed=w)
        else:
            threads = []
            queue = Queue(len(self.input))
            for i in self.input:
                queue.put(i)
            for i in range(self.cores):
                t = threading.Thread(target=learnThread, args=(i, self, queue))
                t.daemon = True
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

    # the Cython version of the model, instantiated on first request, which should be after the vocabulary was build.
    def getModelC(self):
        if not hasattr(self, 'model_c'):
            self.setDebugTarget()
            self.model_c = model_c(self)
        return self.model_c

    # for debugging
    def setDebugTarget(self):
        if self.debugtarget is None:
            self.debugtarget = -1
        elif self.debugtarget == 'all':
            self.debugtarget = -2
        else:
            self.debugtarget = self.vocab.get(self.debugtarget).index

    def learn(self, threadid, feed):
        for module in self.pipeline:
            feed = module(threadid=threadid, model=self, feed=feed)

# a thread that repeatedly pulls a chunk of input data from the queue and calls
# the trainer on that chunk, until there is no more input
def learnThread(threadid, model, queue):
    print("learn", threadid)
    while True:
        try:
            feed = queue.get(False)
        except Empty:
            break
        model.learn(threadid=threadid, feed=feed)

# C version of the model, to allow nogil Cython modules to process efficiently while accessing the model
cdef class model_c:
    def __init__(self, m):
        self.w = allocRP(len(m.solution))
        self.w_input = allocI(len(m.solution))
        self.w_output = allocI(len(m.solution))
        self.layer = allocRP(len(m.solution) * m.cores)
        for l in range(len(m.solution)):
            self.w[l] = toRArray(m.solution[l]);
            self.w_input[l] = m.solution[l].shape[0]
            self.w_output[l] = m.solution[l].shape[1]
        self.matrices = len(m.solution)
        self.progress = toIArray(m.progress)
        self.windowsize = m.windowsize
        self.totalwords = m.vocab.total_words
        self.vocsize = len(m.vocab)
        self.cores = m.cores
        self.alpha = m.start_alpha
        self.iterations = m.iterations
        self.debugtarget = m.debugtarget
        self.sigmoidtable = self.createSigmoidTable()
        self.pipelinec = allocVP(100)                   # pipeline of Cython modules to be processed

    # adds a cyhthon module to the pipeline, note no type checking is done
    cdef void addPipeline(self, void *f):
        print("addPipeline", f == NULL)
        for i in range(100):
            if self.pipelinec[i] == NULL:
                self.pipelinec[i] = f
                break

    # returns the module that comes after ME in the pipeline, or the first if ME is not found and NULL if ME
    # is the last module in the pipeline (the first is
    # usually not stored in the pipeline, since it is called by the last module in the Python pipeline)
    cdef void* getNext(self, void *me):
        for i in range(100):
            if self.pipelinec[i] == me:
                return self.pipelinec[i+1]
            elif self.pipelinec[i] == NULL:
                return self.pipelinec[0]

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
    cdef cREAL *getLayer(self, int thread, int layer) nogil:
        if self.layer[thread * self.cores + layer] == NULL:
            size = self.getLayerSize(layer)
            self.layer[thread * self.cores + layer] = allocR(size)
        return self.layer[thread * self.cores + layer]

    # returns the size of layer #layer
    cdef cINT getLayerSize(self, int layer) nogil:
        return self.w_input[layer] if layer < self.matrices else self.w_output[layer - 1]

    # returns a float that contains is the fraction of words processed, for reporting and to adjust the learning rate
    cdef float getProgress(self) nogil:
        cdef int s = 0
        for i in range(self.cores):
            s += self.progress[i]
        return s / float(self.totalwords)
