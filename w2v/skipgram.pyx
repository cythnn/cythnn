import cython, math
from numpy import float32, int32
from nn cimport saxpy, sdot
import numpy as np
cimport numpy as np
ctypedef np.int32_t cINT
ctypedef np.float32_t cREAL

from libc.string cimport memset

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef float fONE = 1.0
cdef float fZERO = 0.0

cdef cREAL* toRArray(np.ndarray a):
    return <cREAL *>(np.PyArray_DATA(a))

cdef cINT* toIArray(np.ndarray a):
    return <cINT *>(np.PyArray_DATA(a))

def pointtable(vocab, vectorsize, maxdepth):
    print("start pointtable")
    vocsize = len(vocab)
    target = np.zeros((vocsize, maxdepth), dtype=int32)
    exp = np.zeros((vocsize, maxdepth), dtype=int32)
    for word in vocab.values():
        for idx, (l, w) in enumerate(word.innernodes):
            target[word.index][idx] = w.index
            exp[word.index][idx] = l
    print("end pointtable")
    return target, exp

class model:
    def __init__(self, vocab, solution, cores, start_alpha, windowsize):
        self.vocab = vocab
        self.solution = solution
        self.cores = cores
        self.maxdepth = math.log(len(vocab), 2) * 2
        self.windowsize = windowsize
        self.vectorsize = solution.vector_size
        self.inner, self.exp = pointtable(vocab, self.vectorsize, self.maxdepth)
        self.neu1e = np.ndarray((cores, self.vectorsize), dtype=float32)
        self.start_alpha = start_alpha
        self.syn0 = np.ndarray((len(vocab), self.vectorsize), dtype=float32)
        self.syn1 = np.ndarray((len(vocab)-1, self.vectorsize), dtype=float32)
        self.model_c = model_c(self)

cdef class model_c:
    cdef cREAL *syn0, *syn1, *neu1e
    cdef int windowsize, vectorsize, totalwords, cores, maxdepth
    cdef cINT *inner, *exp
    cdef float alpha
    def __init__(self, m):
        self.syn0 = toRArray(m.syn0)
        self.syn1 = toRArray(m.syn1)
        self.neu1e = toRArray(m.neu1e)
        self.windowsize = m.windowsize
        self.vectorsize = m.vectorsize
        self.totalwords = m.vocab.total_words
        self.cores = m.cores
        self.inner = toIArray(m.inner)
        self.exp = toIArray(m.exp)
        self.alpha = m.start_alpha
        self.maxdepth = m.maxdepth

def sg_py(id, model, sentence):
    sg(id, model.model_c, sentence)

cdef void sg(int id, model_c m, np.ndarray sen):
    cdef int sentencelength = len(sen)
    cdef unsigned long long next_random = 1;
    cdef cINT* sentence = toIArray(sen)
    cdef cREAL *neu1e = &(m.neu1e[id])
    cdef int a, b, c, d, l1, l2, last_word, word, innernode, sentence_position
    cdef float f, g, alpha

    if True:
        for sentence_position in range(sentencelength):
            alpha = m.alpha
            word = sentence[sentence_position]
            next_random = next_random * rand + 11
            b = next_random % m.windowsize
            for a in range (b, m.windowsize * 2 + 1 - b):
                if a != m.windowsize:
                    #print("b %d a %d"%(b, a));
                    c = sentence_position - m.windowsize + a
                    if c < 0 or c >= sentencelength: continue
                    last_word = sentence[c]
                    #print("lastword %d"%last_word)
                    l1 = last_word * m.vectorsize
                    memset(neu1e, 0, m.vectorsize * 4)
                    d = 0
                    innernode = m.inner[word * m.maxdepth + d]
                    while True: # over inner nodes of word in output layer
                        #print("innernode %d"%innernode);
                        l2 = innernode * m.vectorsize
                        f = sdot(&m.vectorsize, &(m.syn0[l1]), &ONE, &(m.syn1[l2]), &ONE)
                        g = (1 - m.exp[word * m.maxdepth + d] - f) * alpha
                        saxpy(&m.vectorsize, &g, &(m.syn1[l2]), &ONE, neu1e, &ONE)
                        saxpy(&m.vectorsize, &g, &(m.syn0[l1]), &ONE, &(m.syn1[l2]), &ONE)
                        d += 1
                        innernode = m.inner[word * m.maxdepth + d]
                        if innernode == 0: break
                    #print("break")
                    saxpy(&m.vectorsize, &fONE, neu1e, &ONE, &(m.syn0[l1]), &ONE)



    #memset(work, 0, size * cython.sizeof(REAL_t))
    #for b in range(codelen):
    #     row2 = word_point[b] * size
    #     f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
    #     if f <= -MAX_EXP or f >= MAX_EXP:
    #         continue
    #     f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
    #     g = (1 - word_code[b] - f) * alpha
    #     our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
    #     our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
    #     our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)

