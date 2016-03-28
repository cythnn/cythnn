from multiprocessing import Pool
from tools.worddict import *
from tools.taketime import *
from tools.file import *
from functools import partial

def m(stream):
    return countWords(stream)

@taketime("process")
def process(file, cores = 2, parts = 2, rnge = None):
    pool = Pool(processes=cores)
    wordstreams = wordStreams(file, rnge=rnge)

    tokens = pool.map(m, wordstreams)
    merged = mergeDicts(tokens)
    terms = toList(merged)
    terms = sortTermFreq(terms)
    return terms

if __name__ == '__main__':
    terms = process("../text8", rnge=range(0,1000000), cores=2, parts=4)
    for pair in terms[:20]:
        print(pair[0], ":", pair[1])
