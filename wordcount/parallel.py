import sys
from multiprocessing import Pool
from wordcount import *
from functools import partial

@taketime("process")
def process(path, cores = 2, parts = 2, rnge = None):
    pool = Pool(processes=cores)
    if rnge is None:
        rnge= range(0, size(path))
    map1 = partial(map, path=path)

    tokens = pool.map(map1, chunkRange(rnge, parts))
    tokens_per_term = partitionMap(tokens)
    d = chunkDict(tokens_per_term, parts)
    tokens_reduced = pool.map(reduce, d)
    tokens_flattened = [ i for sub in tokens_reduced for i in sub ]
    terms = sort(tokens_flattened)
    return terms

if __name__ == '__main__':
    terms = process("../text8", rnge=None, cores=4, parts=4)
    for pair in terms[:20]:
        print(pair[0], ":", pair[1])
