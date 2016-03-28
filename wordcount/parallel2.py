import sys
from multiprocessing import Pool
from wordcount2 import *
from functools import partial

@taketime("process")
def process(path, cores = 2, parts = 2, rnge = None):
    pool = Pool(processes=cores)
    if rnge is None:
        rnge= range(0, size(path))
    map1 = partial(map, path=path)

    tokens = pool.map(map1, chunkRange(rnge, parts))
    merged = mergeDicts(tokens[0])
    terms = [(x, y) for x, y in merged]
    terms = sort(terms)
    return terms

if __name__ == '__main__':
    terms = process("../text8", rnge=range(0,1000000), cores=4, parts=4)
    for pair in terms[:20]:
        print(pair[0], ":", pair[1])
