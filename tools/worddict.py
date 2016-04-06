import os, math, re
from queue import Queue

import numpy as np
from collections import Counter, defaultdict
from itertools import islice
from multiprocessing.pool import Pool

from numpy import int32, uint64, int8

# wraps a generator of words read from a flat text #file, within position #byterange
# uses virtual soft partitioning of flat text files, a partition starts after the first whitespace
# and the prior partition reads until the first word seperator after the boundary
from tools.taketime import taketime


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
@taketime("mergedicts")
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

def build_vocab(model):
    pool = Pool(processes=model.cores)
    tokens = pool.map(countWords, model.input)
    merged = mergeDicts(tokens)
    model.vocab = Vocabulary(merged, model.mintf)
    model.outputsize = len(model.vocab)


# reads a stream of words and returns a list of its word id's
# the window of the wordstream is set to match the model, to allow to retrieve #window words
# before and after the designated range. When successful, wentBack and wentPast indicate the
# number of words read before and after the designated range
def readWordIds(threadid, model, feed):
    feed.windowsize = model.windowsize
    def genWords(str, vocab):
        for term in str:
            word = vocab.get(term)
            if word is not None:
                yield word.index

    return np.fromiter(genWords(feed, model.vocab), dtype=int32), feed.wentBack, feed.wentPast

@taketime("createHierarchicalSoftmaxTree")
def createHierarchicalSoftmaxTree(vocab):
    newnodes = []
    pos1 = len(vocab) - 1
    pos2 = 0

    for a in range(len(vocab) - 1):
        if pos1 >= 0:
            if pos2 >= len(newnodes) or vocab[pos1].count < newnodes[pos2].count:
                left = vocab[pos1]
                pos1 -= 1
            else:
                left = newnodes[pos2];
                pos2 += 1
        else:
            left = newnodes[pos2]
            pos2 += 1
        if pos1 >= 0:
            if pos2 >= len(newnodes) or vocab[pos1].count < newnodes[pos2].count:
                right = vocab[pos1]
                pos1 -= 1
            else:
                right = newnodes[pos2]
                pos2 += 1
        else:
            right = newnodes[pos2]
            pos2 += 1
        newnode = Word(left.count + right.count, index=len(vocab) - a - 2)
        left.parent = newnode
        right.parent = newnode
        # print("merge %d -> %d %d"%(newnode.index, left.index, right.index))
        right.isRight = True
        newnodes.append(newnode)

    for word in vocab:
        w = word.parent
        path_length, leftright, nodes = 0, [word.isRight], []
        while hasattr(w, 'index') and w.index > 0:
            nodes.append(w)
            leftright.append(w.isRight)
            w = w.parent
        nodes.append(w)
        leftright.reverse()
        nodes.reverse()
        word.innernodes = [(l, n) for l, n in zip(leftright, nodes)]
        # print(word.word, [ (y.index, int(x)) for (x, y) in word.innernodes ])

@taketime("pointtable")
def pointtable(vocab, vectorsize, maxdepth):
    vocsize = len(vocab)
    target = np.zeros((vocsize, maxdepth), dtype=int32)
    exp = np.zeros((vocsize, maxdepth), dtype=int8)
    for word in vocab.values():
        for idx, (l, w) in enumerate(word.innernodes):
            target[word.index][idx] = w.index
            exp[word.index][idx] = l
    return target, exp


class Vocabulary(defaultdict):
    def __init__(self, vocab, MIN_TF):
        super(Vocabulary, self).__init__()

        vocab = sorted(vocab.items(), key=lambda x: -x[1])

        words = [Word(0, word="</s>", index=0)]
        for word, count in vocab:
            if count >= MIN_TF and word != "</s>":
                words.append(Word(count, word=word, index=len(words)))
        #createHierarchicalSoftmaxTree(words)

        total_words = 0
        for word in words:
            self.__setitem__(word.word, word)
            total_words += word.count
        self.sorted = words
        self.total_words = total_words

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
