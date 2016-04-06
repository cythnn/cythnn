from functools import partial

from numpy import int32, uint64
from model.model cimport *
from tools.worddict import readWordIds
from w2vTrainer.train cimport trainW2V

import numpy as np
cimport numpy as np

cdef cULONGLONG rand = uint64(25214903917)

# An implemenation of a trainerfeeder that processes the input into a list of training
# triples (word, output_position, expected_value). This implementation constructs a
# hierarchical softmax, i.e. a Huffmann tree of the vocabulary, in which iteratively a
# new inner node is added that combines the two nodes with the lowest collection frequency.
# Thus words that occur most frequently obtain shorter paths from the root. The tree is used
# as a replacement for the vocabulary in the output layer, to train the path over nodes to
# reach the predicted word. For this, every inner node has a (virtual) position in the output
# layer, and it's expected value is 0 if the targetted word is in the left subtree or 1 when
# in the right subtree.

def build_hierarchical_softmax(model):
    # short py preparation that sets up a numpy array with the sorted collection frequencies
    # of the words in the vocabulary V. The array is size 2*|V|-1 to also contain the inner nodes.
    # For the words in V, the position corresponds to its index, and </s> is kept in position 0
    model.outputsize = len(model.vocab) - 1
    ctable = np.empty((2 * len(model.vocab) - 1), dtype=int32)
    for i, w in enumerate(model.vocab.sorted):
        ctable[i] = w.count
    build_hierarchical_softmax2(model.getModelC(), ctable)

cdef void build_hierarchical_softmax2(model_c model, ndarray counts):
    cdef cINT *ctable = toIArray(counts)
    cdef int upper = 2 * model.vocsize - 1
    cdef int root = 2 * model.vocsize - 2
    cdef cINT *ptable = allocI(upper)
    cdef cBYTE *rtable = allocB(upper)
    cdef int pos1 = model.vocsize - 1
    cdef int pos2 = model.vocsize
    cdef int maxinner = model.vocsize
    cdef int left, right, pathlength, t

    for maxinner in range(model.vocsize, upper):
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
        ctable[maxinner] = ctable[left] + ctable[right]
        ptable[left] = maxinner
        ptable[right] = maxinner
        rtable[right] = 1
        rtable[left] = 0

    model.innernodes = allocIP(model.vocsize)
    model.exp = allocBP(model.vocsize)

    for w in range(model.vocsize):
        pathlength = 0
        t = w
        while t < root:
            pathlength += 1
            t = ptable[t]
        model.innernodes[w] = allocI(pathlength)
        model.exp[w] = allocB(pathlength)
        pathlength = 0
        t = w
        while t < root:
            model.exp[w][pathlength] = rtable[t]
            t = ptable[t]
            model.innernodes[w][pathlength] = root - t
            pathlength += 1
    free(ptable)
    free(rtable)

def processhs(threadid, model, feed):
    words, wentback, wentpast = feed
    processhs2(threadid, model.model_c, words, wentback, wentpast)

cdef void processhs2(int threadid, model_c m, ndarray words, int wentback, int wentpast):
    cdef cINT *w = toIArray(words)
    cdef int wlength = len(words)
    cdef int windowsize = m.windowsize
    cdef float start_alpha = m.alpha
    cdef cULONGLONG next_random = 1
    cdef cINT *batch = allocI( 100000 * windowsize)
    cdef cINT **innernodes = m.innernodes
    cdef cBYTE **exp = m.exp
    cdef int bindex = 0, word, last_word
    cdef cINT *p_inner
    cdef cBYTE *p_exp
    cdef int threads = m.cores
    cdef int b, i, clower = 0, cupper = 0, wordsprocessed = 0
    cdef followme f = <followme>m.getNext(<void*>processhs2)
    print(f == NULL)

    cdef float alpha = start_alpha * max_float( 1.0 - m.getProgress(), 0.0001 )
    printf("alpha %f\n", alpha)  #if True:
    with nogil:
        for i in range(wlength):
            if i < wentback or i >= wlength - wentpast: continue
            word = w[i]
            if word == 0: continue # end of sentence
            wordsprocessed += 1
            #fsync(fout)
            next_random = next_random * rand + 11;
            b = next_random % windowsize
            clower = i - windowsize + b
            if clower < 0: clower = 0
            #printf("th %d i %d word %d\n", threadid, i, word)
            cupper = i + windowsize + 1 - b
            if cupper > wlength: cupper = wlength
            #fprintf(fout, "%d %d %d\n", b, clower, cupper)
            for c in range(clower, cupper):
                #fprintf(fout, "aap")
                if c != i:
                    last_word = w[c]
                    #fprintf(fout, "%d %d %d\n", last_word, innernodes, &(innernodes[word]))
                    if last_word == 0: break
                    p_inner = innernodes[word]
                    #fprintf(fout, "%d %d\n", p_inner)
                    p_exp = exp[word]
                    #fprintf(fout, "%d %d\n", p_inner, p_exp)
                    #fprintf(fout, "aap %d %d", bindex, 1 if (p_inner == NULL) else 0)
                    while True:
                        batch[bindex] = last_word
                        batch[bindex + 1] = p_inner[0]
                        batch[bindex + 2] = p_exp[0]
                        #printf("sample %d %d %d\n", last_word, p_inner[0], p_exp[0])
                        bindex += 3
                        if p_inner[0] == 0:
                            break
                        p_inner += 1
                        p_exp += 1
                    #printf("bindex %d %d %d\n", bindex, word, last_word)
            #fprintf(fout, "noot\n")
                    #fprintf(fout, "end %d", last_word)
            if bindex > 99000 * windowsize:
                printf("thread %d %d %f\n", threadid, bindex, alpha)
                #for i in range(0, bindex, 3):
                #    printf("%d %d %d\n", batch[i], batch[i+1], batch[i+2])

                f(threadid, m, batch, bindex, alpha)
                printf("a %d", bindex)
                m.progress[threadid] += wordsprocessed
                wordsprocessed = 0
                bindex = 0
                alpha = start_alpha * max_float(1.0 - m.getProgress(), 0.0001)
                printf("alpha %f\n", alpha)
        if bindex > 0:
            #print(threadid, bindex, alpha)
            f(threadid, m, batch, bindex, alpha)
            m.progress[threadid] += wordsprocessed
            #printf("b %d\n", bindex)
            #bindex = 0
            #print("done")
        printf("end %d\n", threadid)

