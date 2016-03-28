from multiprocessing import Pool
from tools.worddict import *
from tools.taketime import *
from tools.file import *
from functools import partial

@taketime("process")
def process(file, cores = 2, parts = 2, byterange = None):
    pool = Pool(processes=cores)
    wordstreams = wordStreams(file, byterange=byterange)

    tokens = pool.map(countWords, wordstreams)
    merged = mergeDicts(tokens)
    terms = toList(merged)
    terms = sortTermFreq(terms)
    return terms

if __name__ == '__main__':
    terms = process("../text8", byterange=range(0, 1000000), cores=2, parts=4)
    for pair in terms[:20]:
        print(pair[0], ":", pair[1])
