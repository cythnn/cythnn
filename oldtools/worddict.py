import os, math
from collections import Counter
from itertools import islice
from oldtools.file import size

# wraps a generator of words read from a flat text #file, within position #byterange
# uses virtual soft partitioning of flat text files, a partition starts after the first whitespace
# and the prior partition reads until the first word seperator after the boundary
class WordStream:
    def __init__(self, byterange=None, file=None):
        self.file = file
        if file and byterange is None:
            self.range = range(0, size(file))
        else:
            self.range = byterange

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

#setup a list of #parts WordStream objects, that cover the given #byterange
def wordStreams(file, parts = 2, byterange = None):
    if byterange is None:
        byterange = range(0, size(file))
    return [WordStream(r, file)
            for r in chunkRange(byterange, parts)]

#split range in #n consecutive sub-ranges
def chunkRange(rnge, n):
    step = math.ceil(len(rnge) / n)
    return [ range(i, min(rnge.stop, i + step))
             for i in range(rnge.start, rnge.stop, step) ]

#split the dictionary in #parts, to prepare for parallel processing
def chunkDict(dict, parts = 2):
    size = math.ceil(len(dict) / parts)
    it = iter(dict.items())
    for i in range(0, len(dict), size):
        yield {k[0]:k[1] for k in islice(it, size)}

#sort term-frequency pairs: freq DESC, term ASC
def sortTermFreq(terms):
    terms.sort(key=lambda x: x[0])
    terms.sort(key=lambda x: -x[1])
    return terms

#merge a list of term-frequency dictionaries
def mergeDicts(dicts):
    first, *rest = dicts
    first = Counter(first)
    for d in rest:
        first += Counter(d)
    return first

#return a term-frequncy dict of the given word-iterable
def countWords(words):
    dict = {}
    for word in words:
        try:
            dict[word] += 1
        except KeyError:
            dict[word] = 1
    return dict

#convert term-freq dictionary to list (for sorting)
def toList(dict):
    return [ (x, y) for x, y in dict.items() ]