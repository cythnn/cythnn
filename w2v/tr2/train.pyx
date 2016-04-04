import cython

from tools.hs.hs import trainerWrapper
from tools.nnmodel.model cimport *
from tools.blas.blas cimport sdot, saxpy
from numpy cimport ndarray
from libc.string cimport memset

cdef int iONE = 1
cdef float fONE = 1

def trainPtr():
    return trainerWrapper().set(train_c)

cdef void train_c(int threadid, cREAL *hiddenlayer, int vectorsize,
                      cREAL *w0, cREAL *w1, cINT *samples,
                      int length, cREAL* exptable, float alpha) nogil:
    cdef int i, last_word, exp, inner, l1, l2, previousword
    cdef cREAL f, g
    #@cython.boundscheck(False)  # turn off bounds-checking for entire function
    #@cython.wraparound(False)  # turn off negative index wrapping for entire function
    #with nogil:

    inner = samples[1]
    for i in range(0, length, 3):
        last_word = samples[i]
        exp = samples[i + 2]
        #print(last_word, inner, exp)
        if (inner == 0):
            memset(hiddenlayer, 0, vectorsize * 4)
        l1 = last_word * vectorsize
        l2 = inner * vectorsize
        #print("f w0 %.10f %.10f" % (w0[l1], w0[l1 + 1]))
        #print("f w1 %.10f %.10f" % (w1[l2], w1[l2 + 1]))
        f = sdot( &vectorsize, &(w0[l1]), &iONE, &(w1[l2]), &iONE)
        if f >= -6 and f <= 6:
            g = exptable[(int)((f + 6) * (1000 / 6 / 2))]
            g = (1 - exp - g) * alpha
            #print("last_word %d inner %d exp %d" % (last_word, inner, exp))
            #print("f %.10f g %.10f" % (f, g))
            #print("l1 %d l2 %d" % (l1, l2))
            #print("syn0 %.10f %.10f" % (w0[l1], w0[l1 + 1]))
            #print("syn1 %.10f %.10f" % (w1[l2], w1[l2 + 1]))
            saxpy( &vectorsize, &g, &(w1[l2]), &iONE, hiddenlayer, &iONE)
            saxpy( &vectorsize, &g, &(w0[l1]), &iONE, &(w1[l2]), &iONE)
        # check if the next sample backpropagates against the root (inner=0)
        # if so we are going to train a different last_word and the
        # hidden layer must be updated to the embedding of the last_word
        inner = samples[i + 4] if i < length - 3 else 0
        if inner == 0:
            saxpy( &vectorsize, & fONE, hiddenlayer, &iONE, &(w0[l1]), &iONE)
            #print("neu %.10f %.10f" % (hiddenlayer[0], hiddenlayer[1]))
            #print("syn0 %.10f %.10f" % (w0[l1], w0[l1 + 1]))
    #print("processed")

