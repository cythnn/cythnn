import os, math, re
from queue import Queue

import numpy as np
from collections import Counter, defaultdict
from itertools import islice
from multiprocessing.pool import Pool
from tools.wordio import inputUniform

from numpy import int32, uint64, int8

# wraps a generator of words read from a flat text #file, within position #byterange
# uses virtual soft partitioning of flat text files, a partition starts after the first whitespace
# and the prior partition reads until the first word seperator after the boundary
from tools.taketime import taketime

# build vocabulary from a stream of words
def buildvocab(learner, model):
    if not hasattr(model, 'vocab'):
        pool = Pool(processes=model.cores)
        chunks = [ model.inputstreamclass(c[0], model.windowsize, c[1]) for c in inputUniform(model.input, model.threads) ]
        tokens = pool.map(countWords, chunks)
        merged = mergeDicts(tokens)
        v = Vocabulary(merged, model.mintf)
        model.setVocab(v)
        if model.quiet == 0:
            print("vocabulary build |v|=%d |c|=%ld"%(len(v), v.totalwords))

def normalize(w1):
    return (w1 / math.sqrt(sum([w * w for w in w1])))

# split the dictionary in #parts, to prepare for parallel processing
def chunkDict(dict, parts=2):
    size = math.ceil(len(dict) / parts)
    it = iter(dict.items())
    for i in range(0, len(dict), size):
        yield {k[0]: k[1] for k in islice(it, size)}


# sort term-frequency pairs: freq DESC, term ASC
def sortTermFreq(terms):
    terms.sort(key=lambda x: x[0])
    terms.sort(key=lambda x: -x[1])
    return terms

# merge a list of term-frequency dictionaries
#@taketime("mergedicts")
def mergeDicts(dicts):
    first, *rest = dicts
    first = Counter(first)
    for d in rest:
        first += Counter(d)
    return first

# return a term-frequncy dict of the given word-iterable
def countWords(words):
    dict = {}
    for word in words:
        try:
            dict[word] += 1
        except KeyError:
            dict[word] = 1
    return dict


# convert term-freq dictionary to list (for sorting)
def toList(dict):
    return [(x, y) for x, y in dict.items()]

# reads a stream of words and returns a list of its word id's
# the window of the wordstream is set to match the model, to allow to retrieve #window words
# before and after the designated range. When successful, wentBack and wentPast indicate the
# number of words read before and after the designated range
@taketime("readWordIds")
def readWordIds(threadid, model, feed):
    feed.windowsize = model.windowsize
    def genWords(str, vocab):
        for term in str:
            word = vocab.get(term)
            if word is not None:
                yield word.index

    return np.fromiter(genWords(feed, model.vocab), dtype=int32), feed.wentBack, feed.wentPast

class Vocabulary(defaultdict):
    def __init__(self, vocab, MIN_TF):
        super(Vocabulary, self).__init__()

        vocab = sorted(vocab.items(), key=lambda x: -x[1])

        words = [Word(0, word="</s>", index=0)]     # add </s> as end-of-sentence character to the first position in the vocabulary
        for word, count in vocab:
            if count >= MIN_TF:
                if word != "</s>":
                    words.append(Word(count, word=word, index=len(words)))
                else:
                    words[0].count = count

        totalwords = 0
        for word in words:
            self.__setitem__(word.word, word)
            if word.index != 0:
                totalwords += word.count
        self.sorted = words
        self.totalwords = totalwords

    def lookup_words(self, sentence):
        result = []
        for word in sentence:
            dict_word = self.get(word)
            if (dict_word != None):
                result.append(dict_word)
        return result

    def lookup_wordids(self, sentence):
        result = []
        for word in sentence:
            dict_word = self.get(word)
            if (dict_word is not None):
                result.append(dict_word.index)
        result = np.array(result, dtype=int32)
        return result

    def lookup_vector(self, term, solution):
        word = self.get(term)
        return solution.syn0[word.index]

    def similarity(self, term, term2, solution):
        return np.dot(self.get(term).getNormalized(solution), self.get(term2).getNormalized(solution))

    # saves the dictionary to file
    def save(self, fname):
        s = sorted(self.vocab.items(), key=lambda x: x[1].index)
        with open(fname, 'w') as fout:
            fout.write("%s\n" % (len(self)))
            for word in self.sorted:
                fout.write("%d %s\n" % (word.index, word.word))

class Word:
    next_random = uint64(1)

    def __init__(self, count, index, word=None):
        self.count = count
        self.isRight = False
        self.parent = None
        if word != None:
            self.word = word
        self.index = index

    def getNormalized(self, solution):
        if not hasattr(self, 'normalized'):
            self.normalized = normalize(solution.syn0[self.index])
        return self.normalized

    def __str__(self):
        if hasattr(self, 'word'):
            return self.word
        else:
            return str(self.index)
