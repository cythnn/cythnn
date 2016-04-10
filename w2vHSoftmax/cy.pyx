from numpy import int32, uint64
from model.cy cimport *
from tools.taketime import taketime

import numpy as np
cimport numpy as np

# This builder constructs a hierarchical softmax, i.e. a Huffmann tree of the vocabulary, in which iteratively a
# new inner node is added that combines the two nodes with the lowest collection frequency.
# Thus words that occur most frequently obtain shorter paths from the root. The tree is used
# as a replacement for the vocabulary in the output layer, to train the path over nodes to
# reach the predicted word. For this, every inner node has a (virtual) position in the output
# layer, and it's expected value is 0 if the targetted word is in the left subtree or 1 when
# in the right subtree.

@taketime("build_hs_tree")
def build_hs_tree(model):
    # short py preparation that sets up a numpy array with the sorted collection frequencies
    # of the words in the vocabulary V. The array is size 2*|V|-1 to also contain the inner nodes.
    # For the words in V, the position corresponds to its index, and </s> is kept in position 0

    if not hasattr(model, 'hs_tree_build'):
        model.outputsize = len(model.vocab) - 1
        ctable = np.empty((2 * len(model.vocab) - 1), dtype=int32)
        for i, w in enumerate(model.vocab.sorted):
            ctable[i] = w.count

        # The tree is stored in the model, allowing multiple modules to access
        build_hs_tree2(model.getModelC(), ctable)
        model.hs_tree_build = True

cdef void build_hs_tree2(modelc model, ndarray counts):
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

    # store tree in the model
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
