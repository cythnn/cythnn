from nqe.findmax import *
import sys
import numpy

if __name__ == "__main__":
    vectorfile = sys.argv[1]
    wordfile = sys.argv[2]
    table, word2vec, vec2word = loadvectors(vectorfile)
    words = loadAnalogy(wordfile, word2vec)
    words = numpy.array([ 1, 2 ])
    a = findmax(words, table)
    print(a)

