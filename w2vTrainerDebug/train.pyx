import cython

from libc.stdio cimport *
from model.model cimport *
from blas.blas cimport sdot, saxpy
from numpy cimport ndarray
from libc.string cimport memset

cdef int iONE = 1
cdef float fONE = 1

# general W2V training module, that trains the solution in the model based on the
# #length samples that are passed. Every sample is a triple of a last_word, for which
# the embedding is learned, a position in the output layer, and the expected value.

#@cython.boundscheck(False)  # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void trainW2V(int threadid, model_c model,
                      cINT *samples, int length, float alpha) nogil:
    cdef int i, j, last_word, exp, inner, l1, l2, previousword
    cdef int vectorsize = model.getLayerSize(1)
    cdef cREAL* sigmoidtable = model.sigmoidtable
    cdef cREAL* w0 = model.w[0]
    cdef cREAL* w1 = model.w[1]
    cdef cREAL* hiddenlayer = model.getLayer(threadid, 1)
    cdef cREAL f, g
    cdef int debugtarget = model.debugtarget

    printf("alpha %f\n", alpha)
    memset(hiddenlayer, 0, vectorsize * 4)
    for i in range(0, length, 3):
        last_word = samples[i]
        inner = samples[i + 1]
        exp = samples[i + 2]
        l1 = last_word * vectorsize
        l2 = inner * vectorsize

        if last_word == debugtarget or debugtarget < -1 or w1[0] > 100:
             printf("th %d last_word %d inner %d exp %d\n", threadid, last_word, inner, exp)
             printf("th %d f w0 %.10f %.10f\n", threadid, w0[l1], w0[l1 + 1])
             printf("th %d f w1 %.10f %.10f\n", threadid, w1[l2], w1[l2 + 1])

        f = sdot( &vectorsize, &(w0[l1]), &iONE, &(w1[l2]), &iONE)

        # if (f < -6):
        #     g = (1 - exp) * alpha
        # elif (f > 6):
        #     g = -exp * alpha
        # else:
        #     g = model.sigmoidtable[ < int > ((f + 6) * (1000 / 6 / 2))]
        #     g = (1 - exp - g) * alpha
        if f >= -model.MAX_SIGMOID and f <= model.MAX_SIGMOID:
            g = model.sigmoidtable[<int>((f + model.MAX_SIGMOID) * (model.SIGMOID_TABLE / model.MAX_SIGMOID / 2))]
            g = (1 - exp - g) * alpha

            if last_word == debugtarget or debugtarget < -1 or w1[0] > 100:
                 printf("th %d last_word %d inner %d exp %d\n", threadid, last_word, inner, exp)
                 printf("th %d f %.10f g %.10f fi %d\n", threadid, f, g, <int>((f + 6) * (1000 / 6 / 2)))
                 printf("th %d l1 %d l2 %d\n", threadid, l1, l2)
                 printf("th %d syn0 %.10f %.10f\n", threadid, w0[l1], w0[l1 + 1])
                 printf("th %d syn1 %.10f %.10f\n", threadid, w1[l2], w1[l2 + 1])
                 printf("th %d h0 %.10f h1 %.10f\n", threadid, hiddenlayer[0], hiddenlayer[1])

            saxpy( &vectorsize, &g, &(w1[l2]), &iONE, hiddenlayer, &iONE)
            saxpy( &vectorsize, &g, &(w0[l1]), &iONE, &(w1[l2]), &iONE)

            if last_word == debugtarget or debugtarget < -1 or w1[0] > 100:
                 printf("th %d usyn1 %.10f %.10f\n", threadid, w1[l2], w1[l2 + 1])
                 printf("th %d uh0 %.10f h1 %.10f\n", threadid, hiddenlayer[0], hiddenlayer[1])

        # check if we backpropagates against the root (inner=0)
        # if so this was the last inner node for last_word and the
        # hidden layer must be updated to the embedding of the last_word
        if inner == 0:
            saxpy( &vectorsize, & fONE, hiddenlayer, &iONE, &(w0[l1]), &iONE)
            memset(hiddenlayer, 0, vectorsize * 4)

            if last_word == debugtarget or debugtarget < -1 or w1[0] > 100:
                 printf("th %d neu %.10f %.10f\n", threadid, hiddenlayer[0], hiddenlayer[1])
                 printf("th %d syn0 %.10f %.10f\n", threadid, w0[l1], w0[l1 + 1])
    printf("processed %d\n", threadid)

