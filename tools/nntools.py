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

def createW2V(model):
    if isinstance(model.vocab, Vocabulary):
        l = createMatrices([len(model.vocab), model.vectorsize, model.outputsize], [rand, zeros])
    else:
        l = createMatrices([model.vocab, model.vectorsize, model.outputsize], [rand, zeros])
    model.solution = l if isinstance(l, list) else [l]

def getVector(model, word):
    return model.solution[0][word.index]

def createMatrices(sizes, init):
    layers = []
    for i in range(len(sizes) - 1):
        if len(init) <= i:
            init.append(None)
        if init[i] is None:
            init[i] = zeros
        layers.append(init[i](sizes[i], sizes[i+1]))
    return layers

def save(fname, model, binary=False):
    s = sorted(model.vocab.items(), key=lambda x: x[1].index)
    if binary:
        with open(fname, 'wb') as fout:
            fout.write(("%s %s\n" % (len(model.vocab), model.solution[-1].shape[0])).encode())
            for word, obj in s:
                row = getVector(model, obj)
                fout.write((word + " ").encode())
                fout.write(struct.pack('%sf' % len(row), *row))
    else:
        print("write flat")
        with open(fname, 'w') as fout:
            fout.write("%s %s\n" % (len(model.vocab), model.solution[-1].shape[0]))
            for word, obj in s:
                row = getVector(model, obj)
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
            if abs(l[a,b]) > 1:
                print("fault w %d %d %f"%(a, b, l[a,b]))
    return l

def uniform(input, output):
    return np.random.uniform(-0.5 / output, 0.5 / output,
                             (input, output), dtype=float32)

def empty(input, output):
    return np.ndarray((input, output), dtype=float32)


def zeros(input, output):
    return np.zeros((input, output), dtype=float32)
