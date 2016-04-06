import cython

from libc.stdio cimport *
from model.model cimport *
import model.model
from blas.blas cimport sdot, saxpy
from libc.string cimport memset

cdef int iONE = 1
cdef float fONE = 1.0

# general W2V training module, that trains the solution in the model based on the
# #length samples that are passed. Every sample is a triple of a last_word, for which
# the embedding is learned, a position in the output layer, and the expected value.

cpdef addTrainW2V(model):
    print("addTrainW2V")
    cdef model_c modelc = model.getModelC()
    modelc.addPipeline(<void*>trainW2V)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void trainW2V(int threadid, int pipelineindex, model_c model,
                      cINT *samples, int length, float alpha) nogil:
    printf("aap\n")
    cdef int i, j, last_word, exp, inner, l1, l2, previousword
    cdef int vectorsize = model.getLayerSize(1)
    cdef cREAL* sigmoidtable = model.sigmoidtable
    cdef cREAL* w0 = model.w[0]
    cdef cREAL* w1 = model.w[1]
    cdef cREAL* hiddenlayer = model.getLayer(threadid, 1)
    cdef cREAL f, g
    cdef int debugtarget = model.debugtarget

    memset(hiddenlayer, 0, vectorsize * 4)                      # initialize hidden layer, to aggregate updates for the current embedding
    for i in range(0, length, 3):                               # stream contains triples: (last_word, inner_node, expected value)
        last_word = samples[i]
        inner = samples[i + 1]
        exp = samples[i + 2]

        l1 = last_word * vectorsize                             # index for last_word in weight matrix w0
        l2 = inner * vectorsize                                 # index to inner node in weight matrix w1

        f = sdot( &vectorsize, &(w0[l1]), &iONE, &(w1[l2]), &iONE)  # energy emitted to inner node

        if f >= -model.MAX_SIGMOID and f <= model.MAX_SIGMOID:  # commonly, when f=0 or f=1 there is nothing to train
            g = model.sigmoidtable[<int>((f + model.MAX_SIGMOID) * (model.SIGMOID_TABLE / model.MAX_SIGMOID / 2))]
            g = (1 - exp - g) * alpha                           # compute the gradient

            saxpy( &vectorsize, &g, &(w1[l2]), &iONE, hiddenlayer, &iONE)   # update the inner node (appears only once in a path)
            saxpy( &vectorsize, &g, &(w0[l1]), &iONE, &(w1[l2]), &iONE)     # add update to hidden layer

        # check if we backpropagates against the root (inner=0)
        # if so this was the last inner node for last_word and the
        # hidden layer must be updated to the embedding of the last_word
        if inner == 0:
            saxpy( &vectorsize, &fONE, hiddenlayer, &iONE, &(w0[l1]), &iONE)
            memset(hiddenlayer, 0, vectorsize * 4)

