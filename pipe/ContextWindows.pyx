import cython

from pipe.cpipe cimport CPipe
from model.solution cimport *
import numpy as np
cimport numpy as np
from numpy import uint64

cdef uLONG rand = uint64(25214903917)
cdef uLONG eleven = uint64(11)

# for an input array of word id's and a given window size, this determines the context that is considered for every
# word position, by giving the lower (inclusive) and upper (exclusive) bounds. Corresponding to the orginal Word2Vec code
# for each position, a symetrical context size is sampled from a uniform distribution from 1 to window_size. The context
# is truncated over sentence boundaries (wordid = 0).
cdef class contextWindow(CPipe):
    def __init__(self, pipeid, learner):
        CPipe.__init__(self, pipeid, learner)
        self.windowsize = self.model.windowsize
        self.random = allocULong(self.model.threads)
        for i in range(self.model.threads):
            self.random[i] = i

    def feed(self, threadid, task):
        words = task.words          # wordids
        wentback = task.wentback    # number of words at start that are only provided as context
        wentpast = task.wentpast    # number of words at end that are only provided as context
        length = task.length        # length of task.words
        clower = np.zeros(length)
        cupper = np.zeros(length)

        self.process(threadid, toIntArray(task.words), toIntArray(clower), toIntArray(cupper), length, wentback, wentpast)

        newtask = task.nextTask()
        newtask.words = words
        newtask.clower = clower     # binding to task object helps garbage collection
        newtask.cupper = cupper
        newtask.length = length
        # when in blockedmode, settings atiteration prevents processing consecutive tasks until the previous iterations was finished
        if self.model.blockedmode:
            newtask.atiteration = task.iteration
        self.addTask(newtask)

    # for every word position, samples a windowsize and stores the start and end position of the context window
    # in the clower and cupper arrays (for reuse and to simplify the learning Pipe)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void process(self, int threadid, cINT *words, cINT *clower, cINT *cupper,
                                         int length, int wentback, int wentpast):
        cdef:
            int b, i, pos, word
            int first_word = 0, next_newline

        with nogil:
            for i in range(wentback):                         # find the first word in the first sentence
                if words[i] == 0:                             # after an optional sentence boundary
                    first_word = i + 1

            pos = wentback                                    # pos is the next word position to try as window center
            while pos < length - wentpast:                    # first_word and next_newline determine sentence boundaries
                next_newline = length
                for i in range(pos, length - wentpast):
                    if words[i] == 0:
                        next_newline = i
                        break;
                if pos < next_newline and next_newline - first_word > 2: # a sentence of size > 1 is found
                    while pos < next_newline and pos < length - wentpast:
                        self.random[threadid] = self.random[threadid] * rand + eleven;   # sample a window size b =[0, windowsize]
                        b = self.random[threadid] % self.windowsize
                        clower[pos] = max_int(first_word, pos - self.windowsize + b) # limit the window with sentence bounaries
                        cupper[pos] = min_int(pos + self.windowsize + 1 - b, next_newline)
                        pos += 1
                pos = next_newline + 1
                first_word = pos
