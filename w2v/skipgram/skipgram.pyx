import cython, math
from numpy import float32, int32
from tools.blas.blas cimport sdot, saxpy
from tools.nnmodel.model cimport *
import numpy as np
from libc.math cimport exp
cimport numpy as np
#ctypedef np.int32_t cINT
#ctypedef np.float32_t cREAL

from libc.string cimport memset
cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cdef unsigned long long rand = 25214903917
cdef int ONE = 1
cdef int ZERO = 0
cdef cREAL fONE = 1.0
cdef cREAL fZERO = 0.0

# cdef cREAL* toRArray(np.ndarray a):
#     return <cREAL *>(np.PyArray_DATA(a))
#
# cdef cINT* toIArray(np.ndarray a):
#     return <cINT *>(np.PyArray_DATA(a))

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
    sg(id, model.model_c, sentence, sentenceboundaries)

cdef sg(int id, model_c m, np.ndarray sen, np.ndarray stend):
    cdef int sentencelength = len(sen)
    cdef int startendlength = len(stend)
    cdef unsigned long long next_random = 1;
    cdef cINT* sentence = toIArray(sen)
    cdef cINT* startend = toIArray(stend)
    cdef cREAL *hiddenlayer = m.getLayer(id, 1)
    cdef int a, b, c, d, l1, l2, last_word, word, innernode, sentence_position, word_batch, start, end
    cdef int iterations = m.iterations
    cdef float f, g
    cdef cREAL *w0 = m.w[0]
    cdef cREAL *w1 = m.w[1]
    cdef float alpha = m.alpha
    cdef int vectorsize = m.w_output[0]
    cdef int windowsize = m.windowsize
    cdef cINT *inner = m.inner
    cdef cINT *exp = m.exp

    #@cython.boundscheck(False)  # turn off bounds-checking for entire function
    #@cython.wraparound(False)  # turn off negative index wrapping for entire function
    #with nogil:
    if True:
        for iteration in range(iterations):
            for se in range(0, startendlength, 2):
                start = startend[se]
                end = startend[se+1]
                for sentence_position in range(start, end):
                    if (sentence_position % 1000 == 0):
                        alpha = m.alpha * (1 - (iteration * sentencelength + sentence_position) / (float)(1 + sentencelength * iterations))
                        if alpha < 0.0001 * m.alpha: alpha = 0.0001 * alpha
                    #alpha = m.alpha
                    #print("alpha %f"%alpha)
                    word = sentence[sentence_position]
                    next_random = next_random * rand + 11
                    b = next_random % windowsize
                    for a in range (b, windowsize * 2 + 1 - b):
                        if a != windowsize:
                            c = sentence_position - windowsize + a
                            if c < start or c >= end: continue
                            last_word = sentence[c]
                            l1 = last_word * vectorsize
                            #print("%d %d %d"%(l1, last_word, m.matrix_input[1]))
                            memset(hiddenlayer, 0, vectorsize * 4)
                            d = 0
                            innernode = m.inner[word * m.maxdepth]
                            if last_word == m.target or m.target == -2:
                                print("f w0 %.10f %.10f"%(w0[l1], w0[l1+1]))
                            while True: # over inner nodes of word in output layer
                                #print(last_word, innernode, exp[word * m.maxdepth + d])
                                l2 = innernode * vectorsize
                                if last_word == m.target or m.target == -2:
                                    print("f w1 %.10f %.10f" % (w1[l2], w1[l2 + 1]))
                                f = sdot(&vectorsize, &(w0[l1]), &ONE, &(w1[l2]), &ONE)
                                if f >= -6 and f <= 6:
                                    g = m.exptable[(int)((f + 6) * (1000 / 6 / 2))]
                                    g = (1 - exp[word * m.maxdepth + d] - g) * alpha
                                    if last_word == m.target or m.target == -2:
                                        print("last_word %d inner %d exp %d"%(last_word, innernode, exp[word * m.maxdepth + d]))
                                        #print("pos %d c %d d %d"%(sentence_position, c, d))
                                        print("f %.10f g %.10f"%(f, g))
                                        print("l1 %d l2 %d"%(l1, l2))
                                        print("syn0 %.10f %.10f"%(w0[l1], w0[l1 + 1]))
                                        print("syn1 %.10f %.10f"%(w1[l2], w1[l2 + 1]))
                                    saxpy(&vectorsize, &g, &(w1[l2]), &ONE, hiddenlayer, &ONE)
                                    saxpy(&vectorsize, &g, &(w0[l1]), &ONE, &(w1[l2]), &ONE)
                                d += 1
                                innernode = inner[word * m.maxdepth + d]
                                if innernode == 0: break
                            saxpy(&vectorsize, &fONE, hiddenlayer, &ONE, &(w0[l1]), &ONE)
                            if last_word == m.target or m.target == -2:
                                print("neu %.10f %.10f"%(hiddenlayer[0], hiddenlayer[1]))
                                print("syn0 %.10f %.10f"%(w0[l1], w0[l1 + 1]))


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

