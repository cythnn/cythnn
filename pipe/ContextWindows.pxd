from pipe.cpipe cimport CPipe
from model.solution cimport *       # defines the cREAL and cINT types

cdef class contextWindow(CPipe):
    cdef:
        int windowsize  # windowsize for Word2Vec
        uLONG *random   # table of random seeds to prevent memory collisions

        # words, wordlength, wentback and wentpassed are the input to be processed.
        # wentback and wentpast indicate the number of words at the beginning and end of the
        # word array that are only provided as context, not as central positions to train words
        # clower and cupper are empty arrays to be sampled with a window of max windowsize, taking
        # sentence boundaries into account.
        void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper,
                     int wlength, int wentback, int wentpast)

