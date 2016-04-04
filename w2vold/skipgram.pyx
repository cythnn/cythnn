import cython, math
from numpy import float32, int32
from nn cimport saxpy, sdot
import numpy as np
from libc.math cimport exp
cimport numpy as np
ctypedef np.int32_t cINT
ctypedef np.float32_t cREAL

from libc.string cimport memset

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0

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
    def __init__(self, vocab, solution, exp_table, cores, start_alpha, windowsize=5, iterations = 5, target=-1):
        self.vocab = vocab
        self.solution = solution
        self.cores = cores
        self.windowsize = windowsize
        self.vectorsize = solution.vector_size
        self.maxdepth = 100
        self.inner, self.exp = pointtable(vocab, self.vectorsize, self.maxdepth)
        self.neu1e = np.ndarray((cores, self.vectorsize), dtype=float32)
        self.start_alpha = start_alpha
        self.syn0 = solution.syn0
        self.syn1 = solution.syn1
        self.exp_table = exp_table
        self.iterations = iterations
        self.target = target
        self.model_c = model_c(self)

cdef class model_c:
    cdef cREAL *syn0, *syn1, *neu1e, *exptable
    cdef int windowsize, vectorsize, totalwords, cores, maxdepth, vocsize, iterations, target
    cdef cINT *inner, *exp
    cdef float alpha
    def __init__(self, m):
        self.syn0 = toRArray(m.syn0)
        self.syn1 = toRArray(m.syn1)
        self.neu1e = toRArray(m.neu1e)
        self.windowsize = m.windowsize
        self.vectorsize = m.vectorsize
        self.totalwords = m.vocab.total_words
        self.vocsize = len(m.vocab)
        self.cores = m.cores
        self.inner = toIArray(m.inner)
        self.exp = toIArray(m.exp)
        self.alpha = m.start_alpha
        self.maxdepth = m.maxdepth
        self.target = m.target
        self.initSyn0()
        self.exptable = toRArray(m.exp_table)
        self.initExp(len(m.exp_table))
        self.iterations = m.iterations

    cdef void initSyn0(self):
        cdef unsigned long long rand = 25214903917
        cdef unsigned long long next_random = 1
        cdef unsigned long long byte2 = 65535
        cdef cREAL div2 = 65536.0
        cdef cREAL half = 0.5
        cdef cREAL *syn0 = self.syn0
        cdef int vecsize = self.vectorsize
        cdef int vocabsize = self.vocsize
        cdef int a, b
        with nogil:
            for a in range(vocabsize):
                for b in range(vecsize):
                    next_random = next_random * rand + 11;
                    syn0[a * vecsize + b] = (((next_random & byte2) / div2) - half) / vecsize

    cdef void initExp(self, tablesize):
        cdef unsigned long long rand = 25214903917
        cdef unsigned long long next_random = 1
        cdef cREAL TABLE_SIZE = tablesize
        cdef cREAL *expTable = self.exptable
        cdef int i
        with nogil:
            for i in range(1000):
                expTable[i] = exp((float)(6 * 2 * i / TABLE_SIZE - 6))
                expTable[i] = expTable[i] / (float)(1 + expTable[i])

def sg_py(id, model, sentence):
    sentenceboundaries = []
    under = 0
    for i in range(len(sentence)):
        if sentence[i] == 0:
            if i > under + 1:
                sentenceboundaries.append(under)
                sentenceboundaries.append(i)
            under = i + 1
    if under < len(sentence) - 1:
        sentenceboundaries.append(under)
        sentenceboundaries.append(len(sentence))
    sentenceboundaries = np.array(sentenceboundaries, dtype = int32)
    #print(sentenceboundaries)
    sg(id, model.model_c, sentence, sentenceboundaries)

cdef void sg(int id, model_c m, np.ndarray sen, np.ndarray stend):
    cdef int sentencelength = len(sen)
    cdef int startendlength = len(stend)
    cdef unsigned long long next_random = 1;
    cdef cINT* sentence = toIArray(sen)
    cdef cINT* startend = toIArray(stend)
    cdef cREAL *neu1e = &(m.neu1e[id])
    cdef int a, b, c, d, l1, l2, last_word, word, innernode, sentence_position, word_batch, start, end
    cdef int iterations = m.iterations
    cdef float f, g
    cdef cREAL *syn0 = m.syn0
    cdef cREAL *syn1 = m.syn1

    cdef float alpha = m.alpha

    #@cython.boundscheck(False)  # turn off bounds-checking for entire function
    #@cython.wraparound(False)  # turn off negative index wrapping for entire function
    #with nogil:
    if True:
        for iteration in range(iterations):
            for se in range(0, startendlength, 2):
                start = startend[se]
                end = startend[se+1]
                #print("%d %d %d"%(se, start, end))
                for sentence_position in range(start, end):
                    if (sentence_position % 1000 == 0):
                        alpha = m.alpha * (1 - (iteration * sentencelength + sentence_position) / (float)(1 + sentencelength * iterations))
                        if alpha < 0.0001 * m.alpha: alpha = 0.0001 * alpha
                    #alpha = m.alpha
                    #print("alpha %f"%alpha)
                    word = sentence[sentence_position]
                    next_random = next_random * rand + 11
                    b = next_random % m.windowsize
                    for a in range (b, m.windowsize * 2 + 1 - b):
                        if a != m.windowsize:
                            c = sentence_position - m.windowsize + a
                            if c < start or c >= end: continue
                            last_word = sentence[c]
                            l1 = last_word * m.vectorsize
                            #print("%d %d %d"%(l1, last_word, m.vectorsize))
                            memset(neu1e, 0, m.vectorsize * 4)
                            d = 0
                            innernode = m.inner[word * m.maxdepth + d]
                            if last_word == m.target or m.target == -2:
                                print("f w0 %.10f %.10f"%(m.syn0[l1], m.syn0[l1+1]))
                            while True: # over inner nodes of word in output layer
                                #print("innernode %d"%innernode);
                                l2 = innernode * m.vectorsize
                                if last_word == m.target or m.target == -2:
                                    print("f w1 %.10f %.10f" % (m.syn1[l2], m.syn1[l2 + 1]))
                                f = sdot(&m.vectorsize, &(syn0[l1]), &ONE, &(syn1[l2]), &ONE)
                                if f >= -6 and f <= 6:
                                    g = m.exptable[(int)((f + 6) * (1000 / 6 / 2))]
                                    g = (1 - m.exp[word * m.maxdepth + d] - g) * alpha
                                    if last_word == m.target or m.target == -2:
                                        print("pos %d c %d d %d"%(sentence_position, c, d))
                                        print("f %.10f g %.10f"%(f, g))
                                        print("l1 %d l2 %d"%(l1, l2))
                                        print("syn0 %.10f %.10f"%(syn0[l1], syn0[l1 + 1]))
                                        print("syn1 %.10f %.10f"%(syn1[l2], syn1[l2 + 1]))
                                    saxpy(&m.vectorsize, &g, &(syn1[l2]), &ONE, neu1e, &ONE)
                                    saxpy(&m.vectorsize, &g, &(syn0[l1]), &ONE, &(syn1[l2]), &ONE)
                                d += 1
                                innernode = m.inner[word * m.maxdepth + d]
                                if innernode == 0: break
                            saxpy(&m.vectorsize, &fONE, neu1e, &ONE, &(syn0[l1]), &ONE)
                            if last_word == m.target or m.target == -2:
                                print("neu %.10f %.10f"%(neu1e[0], neu1e[1]))
                                print("syn0 %.10f %.10f"%(syn0[l1], syn0[l1 + 1]))


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

