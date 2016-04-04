from multiprocessing.pool import Pool

import cython
from numpy import float32, int32, int8, uint64

from tools.blas.blas cimport sdot, saxpy
from tools.nnmodel.model cimport *
from libc.stdlib cimport free
from w2v.tr2.train cimport *

import numpy as np
cimport numpy as np

cdef cULONGLONG rand = uint64(25214903917)

cdef class trainerWrapper:
    cdef trainerWrapper set(self, train t):
        self.f = t
        return self

class createHS():
    def __init__(self, vocab):
        self.vocab = vocab
        ctable = np.empty((2 * len(vocab) - 1), dtype=int32)
        for i, w in enumerate(vocab.sorted):
            #print(type(w), type(ctable))
            ctable[i] = w.count
        self.hs = HS(ctable)

    def stream(self, model, wordstreams, trainer):
        self.hs.setTr(trainer)
        if (len(wordstreams) == 1):
            self.substream( (0, model, wordstreams[0]) )
        else :
            pool = Pool(processes=model.cores)
            tasks = [ ( i, model, w ) for i, w in enumerate(wordstreams) ]
            pool.map( self.substream, tasks)

    def substream(self, param):
        id, model, wordstream = param
        next_random = uint64(1);
        words = np.fromiter(self.genWords(wordstream), dtype=int32)
        #print(type(words))
        self.hs.stream(id, model, words, wordstream.wentBack, wordstream.wentPast)

    def genWords(self, str):
        print(type(str))
        for term in str:
            word = self.vocab.get(term)
            if word is not None:
                yield word.index

cdef class HS:
    def __init__(self, ndarray counts):
        cdef cINT *ctable = toIArray(counts)
        self.vocsize = len(counts) / 2 + 1
        cdef int upper = 2 * self.vocsize - 1
        cdef int root = 2 * self.vocsize - 2
        cdef cINT *ptable = allocI(upper)
        cdef cBYTE *rtable = allocB(upper)
        cdef int pos1 = self.vocsize - 1
        cdef int pos2 = self.vocsize
        cdef int maxinner = self.vocsize
        cdef int left, right, pathlength, t

        self.innernodes = allocIP(self.vocsize)
        self.exp = allocIB(self.vocsize)

        for maxinner in range(self.vocsize, upper):
            if pos1 >= 0:
                if pos2 >= maxinner or ctable[pos1] < ctable[pos2]:
                    left = pos1
                    pos1 -= 1
                else:
                    left = pos2;
                    pos2 += 1
            else:
                left = pos2
                pos2 += 1
            if pos1 >= 0:
                if pos2 >= maxinner or ctable[pos1] < ctable[pos2]:
                    right = pos1
                    pos1 -= 1
                else:
                    right = pos2
                    pos2 += 1
            else:
                right = pos2
                pos2 += 1
            #print("maxinner %d left %d right %d"%(maxinner, left, right))
            ctable[maxinner] = ctable[left] + ctable[right]
            ptable[left] = maxinner
            ptable[right] = maxinner
            rtable[right] = 1
            rtable[left] = 0


        for w in range(self.vocsize):
            pathlength = 0
            t = w
            while t < root:
                pathlength += 1
                t = ptable[t]
            self.innernodes[w] = allocI(pathlength)
            self.exp[w] = allocB(pathlength)
            pathlength = 0
            t = w
            while t < root:
                self.exp[w][pathlength] = rtable[t]
                t = ptable[t]
                self.innernodes[w][pathlength] = root - t
                #print(w, t, self.innernodes[w][pathlength], self.exp[w][pathlength])
                pathlength += 1
        # for i in range(self.vocsize):
        #     p = 0
        #     d = "word#%d cf#%d "%(i, ctable[i])
        #     while True:
        #         d += "(%d, %d)"%(self.innernodes[i][p], self.exp[i][p])
        #         if self.innernodes[w][p] == 0:
        #             break
        #         p += 1
        #     print(d)
        free(ptable)
        free(rtable)


    def setTr(self, w):
        self.setTrainer(w)

    cdef void setTrainer(self, trainerWrapper w):
        self.f = w.f
        #print("setTrainer", w, w.f is NULL, self.f is NULL)

    def stream(self, model, words, wentback, wentpast):
        pool = Pool(processes=model.cores)
        [ ]

        self.i_stream(threadid, model.model_c, words, wentback, wentpast)


    def stream1(self, params):
        threadid, model, words, wentback, wentpast = params
        self.i_stream(threadid, model.model_c, words, wentback, wentpast)


def parallel(model, wordstreams):

    tokens = pool.map(countWords, [(wordstreams)




    cdef void i_stream(self, int threadid, model_c m, ndarray words, int wentback, int wentpast):
        cdef cINT *w = toIArray(words)
        cdef int wlength = len(words)
        cdef int windowsize = m.windowsize
        cdef float alpha = m.alpha, start_alpha = m.alpha
        cdef cULONGLONG next_random = 1
        cdef cINT *batch = allocI( 100000 * windowsize)
        cdef cINT **innernodes = self.innernodes
        cdef cBYTE **exp = self.exp
        cdef int bindex, word, last_word
        cdef cINT *p_inner
        cdef cBYTE *p_exp
        cdef train tr = self.f
        cdef int threads = m.cores
        cdef int vectorsize = m.vectorsize
        cdef cREAL *hiddenlayer = m.getLayer(threadid, 1)
        cdef cREAL *w0 = m.w[0]
        cdef cREAL *w1 = m.w[1]
        cdef cREAL *exptable = m.exptable
        cdef int b, i, clower, cupper

        with nogil:
        #print("windowsize", windowsize)
            for i in range(wlength):
                if i < wentback or i >= wlength - wentpast: continue
                word = w[i]
                next_random = next_random * rand + 11;
                b = next_random % windowsize
                clower = i - windowsize + b
                if clower < 0: clower = 0
                cupper = i + windowsize + 1 - b
                if cupper > wlength: cupper = wlength
                for c in range(clower, cupper):
                    if c != i:
                        last_word = w[c]
                        if last_word == 0: break
                        p_inner = innernodes[word]
                        p_exp = exp[word]
                        while True:
                            batch[bindex] = last_word
                            batch[bindex + 1] = p_inner[0]
                            batch[bindex + 2] = p_exp[0]
                            bindex += 3
                            if p_inner[0] == 0:
                                break
                            p_inner += 1
                            p_exp += 1
                if bindex > 99000 * windowsize:
                    #print(threadid, vectorsize, bindex, alpha)
                    tr(threadid, hiddenlayer, vectorsize, w0, w1, batch, bindex, exptable, alpha)
                    #print("a")
                    bindex = 0
                    alpha = start_alpha * (wlength - i) / (float)(wlength)
                    if alpha < 0.0001 * start_alpha: alpha = 0.0001 * start_alpha
            if bindex > 0:
                #print(threadid, bindex, alpha)
                tr(threadid, hiddenlayer, vectorsize, w0, w1, batch, bindex, exptable, alpha)
                bindex = 0
                #print("done")

