from pipe.cy cimport cypipe
cimport numpy as np

ctypedef np.uint8_t cBYTE
ctypedef np.int32_t cINT
ctypedef np.float32_t cREAL

cdef class trainCbowHS(cypipe):
    cdef cINT **innernodes  # points to the HS tree
    cdef cBYTE **expected
    cdef int wordsprocessed     # counts the number of processed words, to update the learning rate
    cdef int vectorsize     # size of the hidden layer (=size of the learned embeddings)

    # convenient lookup table for sigmoid function
    cdef int MAX_SIGMOID, SIGMOID_TABLE
    cdef cREAL *sigmoidtable

    # shared weight matrices w0 and w1 and
    # two thread-safe vector for the hidden layer for the feed_forward step (fw) and the back propagation step (bw)
    cdef cREAL *w0, *w1, *hiddenlayer_fw, *hiddenlayer_bw

    cdef int debugtarget  #for debugging purposes

    cdef void process(self, cINT *words, cINT *clower, cINT *cupper, int length) nogil
