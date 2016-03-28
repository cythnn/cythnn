import os, math

from tools.taketime import taketime
from itertools import islice

def size(path):
    return os.path.getsize(path)

# virtual soft partitioning of flat text files, a partition starts after the first whitespace
@taketime("load")
def loadWords(r, path):
    with open(path, "r", buffering=100000) as f:
        words = []
        word = ""
        if r.start > 0:
            f.seek(r.start)
            while True:
                c = f.read(1)
                if len(c) == 0 or c == ' ' or c == '\n':
                    break
        while True:
            c = f.read(1)
            if len(c) == 0:
                break
            if c == ' ' or c == '\n':
                if len(word) > 0:
                    words.append(word)
                word = ""
                if f.tell() > r.stop:
                    return words
            else:
                word += c
        return words

def map(r, path):
    words = loadWords(r, path)
    return [(word, 1) for word in words]

def chunkRange(rnge, n):
    step = math.ceil(len(rnge) / n)
    return [ range(i, min(rnge.stop, i + step))
             for i in range(rnge.start, rnge.stop, step) ]

def chunkDict(d, parts = 2):
    size = math.ceil(len(d) / parts)
    it = iter(d.items())
    for i in range(0, len(d), size):
        yield {k[0]:k[1] for k in islice(it, size)}

@taketime("part")
def partition(L):
    tf = {}
    for p in L:
        try:
            tf[p[0]].append(p)
        except KeyError:
            tf[p[0]] = [p]
    return tf

@taketime("part")
def partitionMap(mappings):
    tf = {}
    for mapping in mappings:
        for tuple in mapping:
            try:
                tf[tuple[0]].append(tuple)
            except KeyError:
                tf[tuple[0]] = [tuple]
    return tf

def reduce(mappings):
    return [ (x, sum( ll[1] for ll in l )) for x, l in mappings.items() ]

def sort(terms):
    terms.sort(key=lambda x: x[0])
    terms.sort(key=lambda x: -x[1])
    return terms

@taketime("process")
def process(path):
    r = range(0, 1000000)
    single_count_tuples = map(r, path)
    token_to_tuples = partition(single_count_tuples)
    term_frequencies = reduce(token_to_tuples)
    terms = sort(term_frequencies)
    return terms

if __name__ == '__main__':
    terms = process("../text8")
    for pair in terms[:20]:
        print(pair[0], ":", pair[1])
