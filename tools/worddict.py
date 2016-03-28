import os, math
from collections import Counter
from itertools import islice
from tools.file import size

# virtual soft partitioning of flat text files, a partition starts after the first whitespace
class WordStream:
    def __init__(self, rnge=None, file=None):
        self.file = file
        if file and rnge is None:
            self.range = range(0, size(file))
        else:
            self.range = rnge

    def __iter__(self):
        with open(self.file, "r", buffering=100000) as f:
            word = ""
            if self.range.start > 0:
                f.seek(self.range.start)
                while True:
                    char = f.read(1)
                    if len(char) == 0 or char == ' ' or char == '\n':
                        break
            while True:
                char = f.read(1)
                if len(char) == 0:
                    break
                if char == ' ' or char == '\n':
                    if len(word) > 0:
                        yield word
                    word = ""
                    if f.tell() > self.range.stop:
                        if len(word) > 0:
                            yield word
                        break
                else:
                    word += char

def wordStreams(file, parts = 2, rnge = None):
    if rnge is None:
        rnge = size(file)
    return [ WordStream(r, "../text8")
            for r in chunkRange(rnge, parts)]

def chunkRange(rnge, n):
    step = math.ceil(len(rnge) / n)
    return [ range(i, min(rnge.stop, i + step))
             for i in range(rnge.start, rnge.stop, step) ]

def chunkDict(dict, parts = 2):
    size = math.ceil(len(dict) / parts)
    it = iter(dict.items())
    for i in range(0, len(dict), size):
        yield {k[0]:k[1] for k in islice(it, size)}

def sortTermFreq(terms):
    terms.sort(key=lambda x: x[0])
    terms.sort(key=lambda x: -x[1])
    return terms

def mergeDicts(dicts):
    first, *rest = dicts
    print(type(dicts[0]))
    first = Counter(first)
    for d in rest:
        first += Counter(d)
    return first

def countWords(words):
    dict = {}
    for word in words:
        try:
            dict[word] += 1
        except KeyError:
            dict[word] = 1
    return dict

def toList(dict):
    return [ (x, y) for x, y in dict.items() ]