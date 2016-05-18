from numpy import int32, uint64
from model.solution cimport *
from tools.taketime import taketime
from libc.stdio cimport *
import heapq, math
from tools.ctypes cimport *
import numpy as np
cimport numpy as np
from queue import PriorityQueue

# This builder constructs a hierarchical softmax, i.e. a Huffmann tree of the vocabulary, in which iteratively a
# new inner node is added that combines the two nodes with the lowest collection frequency.
# Thus words that occur most frequently obtain shorter paths from the root. The tree is used
# as a replacement for the vocabulary in the output layer, to train the path over nodes to
# reach the predicted word. For this, every inner node has a (virtual) position in the output
# layer, and it's expected value is 0 if the targetted word is in the left subtree or 1 when
# in the right subtree.

#@taketime("build_hs_tree")
def hsoftmax(learner, model):
    # short py preparation that sets up a numpy array with the sorted collection frequencies
    # of the words in the vocabulary V. The array is size 2*|V|-1 to also contain the inner nodes.
    # For the words in V, the position corresponds to its index, and </s> is kept in position 0
    if not hasattr(model, 'hs_tree_build'):
        solution = model.getSolution()
        build(model)
        model.outputsize = model.vocsize - 1

cdef void build(object model):
    cdef:
        cINT *ctable = allocInt(model.vocsize * 2)   # temp table for the collection frequency of each node
        cINT *ptable = allocInt(model.vocsize * 2)   # pointer for each node to its parent node
        cBYTE *rtable = allocByte(model.vocsize * 2)  # for each node, if it is reach by taking a left (0) or right (1) turn from its parent in the tree
        Solution solution = model.getSolution()

    # the Huffmann tree is stored in the solution, to allow access to Cython modules
    solution.innernodes = allocIntP(model.vocsize)
    solution.exp = allocByteP(model.vocsize)

    for i, w in enumerate(model.vocab.sorted):
        ctable[i] = w.count

    build2(model, ctable, ptable, rtable, 0, model.vocsize, 1, 1, model.vocsize)

    free(ptable)
    free(rtable)
    free(ctable)

    # The tree is stored in the model, allowing multiple modules to access
    model.hs_tree_build = True

cdef void build2(object model, cINT *ctable, cINT *ptable, cBYTE *rtable, int taskid, int inneroffset, float wordfactor, int start, int end):
    cdef:
        int upper = inneroffset + (end - start) - 1
        Solution solution = model.getSolution()
        int root = upper - 1
        int realroot = model.vocsize * 2 - 3
        int pathlength, t, i

    tree(start, end, inneroffset, wordfactor, ctable, ptable, rtable)

    for w in range(start, end):
        pathlength = 0
        t = w
        while t < root:
            pathlength += 1
            t = ptable[t]
        solution.innernodes[w] = allocInt(pathlength)
        solution.exp[w] = allocByte(pathlength)
        pathlength = 0
        t = w
        while t < root:
            solution.exp[w][pathlength] = rtable[t]
            t = ptable[t]
            solution.innernodes[w][pathlength] = realroot - t
            pathlength += 1

cdef void tree(int start, int end, int inneroffset, float wordfactor, cINT *ctable, cINT *ptable, cBYTE *rtable):
    cdef int upper = inneroffset + (end - start) - 1
    cdef int root = upper - 1
    cdef int pos1 = end - 1
    cdef int pos2 = inneroffset
    cdef int maxinner
    cdef int left, right

    for maxinner in range(inneroffset, upper):
        if pos1 >= start:
            if pos2 >= maxinner or ctable[pos1] <= wordfactor * ctable[pos2]:
                left = pos1
                pos1 -= 1
            else:
                left = pos2;
                pos2 += 1
        else:
            left = pos2
            pos2 += 1
        if pos1 >= start:
            if pos2 >= maxinner or ctable[pos1] <= wordfactor * ctable[pos2]:
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
