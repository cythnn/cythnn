import os, math

from tools.taketime import taketime
from tools.worddict import *

@taketime("process")
def process(path):
    r = range(0, 1000000)
    wordstreams = wordStreams(path, byterange=range(0, 1000000), parts=1)
    counts = countWords(wordstreams[0])
    terms = toList(counts)
    terms = sortTermFreq(terms)
    return terms

if __name__ == '__main__':
    terms = process("../text8")
    for pair in terms[:20]:
        print(pair[0], ":", pair[1])
