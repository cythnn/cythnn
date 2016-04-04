import os, math, re
import numpy as np
import struct

from numpy import int32, uint64, float32

# wraps a generator of words read from a flat text #file, within position #byterange
# uses virtual soft partitioning of flat text files, a partition starts after the first whitespace
# and the prior partition reads until the first word seperator after the boundary
from tools.taketime import taketime
from tools.worddict import Vocabulary, Word


def normalize(w1):
   return (w1 / math.sqrt(sum([ w * w for w in w1 ])))

def createW2V(vocab, vector_size):
    if isinstance(vocab, Vocabulary):
        l = createMatrices([len(vocab), vector_size, len(vocab) - 1], [rand, zeros])
    else:
        l = createMatrices([vocab, vector_size, vocab - 1], [rand, zeros])
    return Solution(l)

def createMatrices(sizes, init):
    layers = []
    for i in range(len(sizes) - 1):
        if len(init) <= i:
            init.append(None)
        if init[i] is None:
            init[i] = zeros
        layers.append(init[i](sizes[i], sizes[i+1]))
    return layers

class Solution:
    MAX_EXP = 6
    EXP_TABLE = 1000

    def __init__(self, matrix):
        self.matrix = matrix if isinstance(matrix, list) else [matrix]
        self.expTable = self.initExpTable()

    def initExpTable(self):
        # fout = open("log", "w")
        table = np.empty((self.EXP_TABLE), float32)
        for i in range(0, 1000):
            e = math.exp(float32(2 * self.MAX_EXP * i / self.EXP_TABLE - self.MAX_EXP))
            table[i] = e / float32(e + 1)
            # fout.write("%d %.10f\n"%(i, table[i]))
        # fout.close()
        return table

def save(fname, vocab, solution, binary=False):
    s = sorted(vocab.items(), key=lambda x: x[1].index)
    if binary:
        with open(fname, 'wb') as fout:
            fout.write(("%s %s\n" % (len(vocab), solution.matrix[-1].shape[0])).encode())
            for word, obj in s:
                row = obj.getVector(solution)
                fout.write((word + " ").encode())
                fout.write(struct.pack('%sf' % len(row), *row))
    else:
        print("write flat")
        with open(fname, 'w') as fout:
            fout.write("%s %s\n" % (len(vocab), solution.matrix[-1].shape[0]))
            for word, obj in s:
                row = obj.getVector(solution)
                fout.write("%s %d %s\n" % (word, obj.count, ' '.join("%f" % val for val in row)))


def load(fname, binary=False, normalized=False):
    with open(fname, 'r') as fin:
        header = fin.readline()
        wordcount, vector_size = [ int(x) for x in header.split(" ") ]
        solution = createW2V(wordcount, vector_size)
        vocab = Vocabulary({}, MIN_TF=-1)
        index = 0
        for line in fin.readlines():
            terms = line.split(" ")
            word = terms[0]
            count = int(terms[1])
            solution.matrix[0][index] = [float32(terms[i]) for i in range(2, len(terms))]
            if normalized:
                solution.matrix[0][index] = normalize(solution.matrix[0][index])
            vocab[word] = Word(count, index=index, word=word)
            index+=1
            vocab.total_words += count
        return vocab, solution

def rand(input, output):
    l = np.empty((input, output), dtype=float32)
    next_random = uint64(1)
    for a in range(input):
        for b in range(output):
            with np.errstate(over='ignore'):
                next_random = next_random * uint64(25214903917) + uint64(11);
            l[a, b] = ((next_random & uint64(65535)) / float32(65536) - 0.5) / output
    return l

def uniform(input, output):
    return np.random.uniform(-0.5 / output, 0.5 / output,
                             (input, output), dtype=float32)

def empty(input, output):
    return np.ndarray((input, output), dtype=float32)


def zeros(input, output):
    return np.zeros((input, output), dtype=float32)
