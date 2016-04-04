import cython, math
from numpy import float32, int32
from tools.blas.blas cimport sdot, saxpy
from tools.worddict import pointtable

import numpy as np
cimport numpy as np

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0

class model:
    def __init__(self, vocab, solution, cores, start_alpha, windowsize=5, iterations = 5, threads = 1, hs = 1, target=-1):
        self.vocab = vocab
        self.solution = solution
        self.cores = cores
        self.windowsize = windowsize
        #print("model", self.windowsize)
        self.threads = threads
        self.maxdepth = 100
        self.start_alpha = start_alpha
        self.iterations = iterations
        self.hs = hs
        self.target = target
        #if hs:
        #    self.inner, self.exp = pointtable(vocab, solution.matrix[-1].shape[0], self.maxdepth)
        self.model_c = model_c(self)

cdef class model_c:
    # cdef cREAL **layer, **update, *exptable
    # cdef int windowsize, vectorsize, totalwords, cores, maxdepth, vocsize, iterations
    # cdef cINT *inner, *exp, *layer_input, *layer_output
    # cdef float alpha

    def __init__(self, m):
        self.w = allocRP(len(m.solution.matrix))
        self.w_input = allocI(len(m.solution.matrix))
        self.w_output = allocI(len(m.solution.matrix))
        self.layer = allocRP(len(m.solution.matrix) * m.threads)
        for l in range(len(m.solution.matrix)):
            self.w[l] = toRArray(m.solution.matrix[l]);
            self.w_input[l] = m.solution.matrix[l].shape[0]
            self.w_output[l] = m.solution.matrix[l].shape[1]
        self.matrices = len(m.solution.matrix)
        self.windowsize = m.windowsize
        self.totalwords = m.vocab.total_words
        self.vocsize = len(m.vocab)
        self.cores = m.cores
        self.alpha = m.start_alpha
        self.maxdepth = m.maxdepth
        self.threads = m.threads
        self.exptable = toRArray(m.solution.expTable)
        #self.initExp(len(m.solution.expTable))
        self.iterations = m.iterations
        self.target = m.target
        #if m.hs:
        #    self.exp = toIArray(m.exp)
        #    self.inner = toIArray(m.inner)

    cdef cREAL *getLayer(self, int thread, int layer):
        if self.layer[thread * self.threads + layer] == NULL:
            size = self.w_input[layer] if layer < self.matrices else self.w_output[layer - 1]
            self.layer[thread * self.threads + layer] = allocR(size)
        return self.layer[thread * self.threads + layer]

    # cdef void initSyn0(self):
    #     cdef unsigned long long rand = 25214903917
    #     cdef unsigned long long next_random = 1
    #     cdef unsigned long long byte2 = 65535
    #     cdef cREAL div2 = 65536.0
    #     cdef cREAL half = 0.5
    #     cdef cREAL *syn0 = self.syn0
    #     cdef int vecsize = self.vectorsize
    #     cdef int vocabsize = self.vocsize
    #     cdef int a, b
    #     with nogil:
    #         for a in range(vocabsize):
    #             for b in range(vecsize):
    #                 next_random = next_random * rand + 11;
    #                 syn0[a * vecsize + b] = (((next_random & byte2) / div2) - half) / vecsize
    #
    # cdef void initExp(self, tablesize):
    #     cdef unsigned long long rand = 25214903917
    #     cdef unsigned long long next_random = 1
    #     cdef cREAL TABLE_SIZE = tablesize
    #     cdef cREAL *expTable = self.exptable
    #     cdef int i
    #     with nogil:
    #         for i in range(1000):
    #             expTable[i] = exp((float)(6 * 2 * i / TABLE_SIZE - 6))
    #             expTable[i] = expTable[i] / (float)(1 + expTable[i])

