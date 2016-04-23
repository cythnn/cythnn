from numpy import int32, uint64
from model.solution cimport *
from tools.taketime import taketime
from libc.stdio cimport *
import heapq, math
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
def build_hs_tree(learner, model):
    # short py preparation that sets up a numpy array with the sorted collection frequencies
    # of the words in the vocabulary V. The array is size 2*|V|-1 to also contain the inner nodes.
    # For the words in V, the position corresponds to its index, and </s> is kept in position 0
    if not hasattr(model, 'hs_tree_build'):
        model.outputsize = model.vocsize - 1
        ctable = np.empty((2 * model.vocsize - 1), dtype=int32)
        for i, w in enumerate(model.vocab.sorted):
            ctable[i] = w.count

        # The tree is stored in the model, allowing multiple modules to access
        build_hs_tree2(model, ctable)
        model.hs_tree_build = True

cdef void build_hs_tree2(object model, ndarray counts):
    cdef int upper = 2 * model.vocsize - 1
    cdef Solution solution = model.getSolution()
    cdef cINT *ctable = toIArray(counts)
    cdef cINT *ptable = allocI(upper)
    cdef cBYTE *rtable = allocB(upper)
    cdef int root = 2 * model.vocsize - 2
    cdef int pathlength, t, i

    tree2(model, ctable, ptable, rtable)

    # store tree in the model
    solution.innernodes = allocIP(model.vocsize)
    solution.exp = allocBP(model.vocsize)
    if model.split:
        word2taskid(model, solution, ptable, ctable)

    for w in range(model.vocsize):
        pathlength = 0
        t = w
        while t < root:
            pathlength += 1
            t = ptable[t]
        solution.innernodes[w] = allocI(pathlength)
        solution.exp[w] = allocB(pathlength)
        pathlength = 0
        t = w
        while t < root:
            solution.exp[w][pathlength] = rtable[t]
            t = ptable[t]
            solution.innernodes[w][pathlength] = root - t
            pathlength += 1

    free(ptable)
    free(rtable)

cdef void tree(object model, cINT *ctable, cINT *ptable, cBYTE *rtable):
    cdef int upper = 2 * model.vocsize - 1
    cdef int root = 2 * model.vocsize - 2
    cdef int pos1 = model.vocsize - 1
    cdef int pos2 = model.vocsize
    cdef int maxinner = model.vocsize
    cdef int left, right

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

cdef void tree2(object model, cINT *ctable, cINT *ptable, cBYTE *rtable):
    cdef int upper = 2 * model.vocsize - 1
    cdef int root = 2 * model.vocsize - 2
    cdef int pos1 = model.vocsize - 1
    cdef int pos2 = model.vocsize
    cdef int maxinner = model.vocsize
    cdef int left, right
    cdef float wordfactor = 1.3

    for maxinner in range(model.vocsize, upper):
        if pos1 >= 0:
            if pos2 >= maxinner or ctable[pos1] < wordfactor * ctable[pos2]:
                left = pos1
                pos1 -= 1
            else:
                left = pos2;
                pos2 += 1
        else:
            left = pos2
            pos2 += 1
        if pos1 >= 0:
            if pos2 >= maxinner or ctable[pos1] < wordfactor * ctable[pos2]:
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


cdef void word2taskid(object model, Solution solution, cINT *ptable, cINT *ctable):
    cdef int i, w, inner
    cdef int root = 2 * model.vocsize - 2
    assignments = PriorityQueue(model.tasks)
    for i in range(model.tasks):
        assignments.put((0, 0, set()))
    assigned = set()

    for w in range(model.vocsize):
        freq, count, nodes = assignments.get()
        count += 1
        i = w
        while i < root:
            i = ptable[i]
            inner = root - i
            if not assigned.__contains__(inner):
                nodes.add(inner)
                assigned.add(inner)
                freq += ctable[i]
            else:
                break
        assignments.put((freq, count, nodes))

    cdef cINT *ia = allocI(model.vocsize - 1)
    cdef int index = 0, singlevolume = 0
    cdef long singles = 0
    for index in range(model.tasks):
         freq, count, nodes = assignments.get()
         if count == 1:
             singles += 1
             singlevolume += freq
         #print(str(nodes))
         for i in nodes:
            ia[i] = index
    singles = 0
    solution.singletaskids = singles
    solution.word2taskid = ia
    cdef long totalwords
    cdef long iterations = model.iterations
    cdef long threads = model.threads
    cdef long vocabwords = model.vocab.totalwords
    totalwords = vocabwords * iterations * (threads - singles)
    totalwords += singles * vocabwords
    solution.totalwords =  totalwords


    printf("totalwords %ll", solution.totalwords)

