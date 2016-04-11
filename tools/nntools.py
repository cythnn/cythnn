import os, math, re
import numpy as np
import struct

from numpy import int32, uint64, float32

# wraps a generator of words read from a flat text #file, within position #byterange
# uses virtual soft partitioning of flat text files, a partition starts after the first whitespace
# and the prior partition reads until the first word seperator after the boundary
from matrix.cy import createMatrices
from tools.taketime import taketime
from tools.worddict import Vocabulary, Word, build_vocab

# normalize a vector (commonly used before comparing vectors)
def normalize(w1):
   return (w1 / math.sqrt(sum([ w * w for w in w1 ])))

# creates a solution space for a word2vec model, 2 weight matrices, initialized resp. random and with zeros
@taketime("createW2V")
def createW2V(model):
    if isinstance(model.vocab, Vocabulary):
        l = createMatrices([len(model.vocab), model.vectorsize, model.outputsize], [2, 0])
    else:
        l = createMatrices([model.vocab, model.vectorsize, model.outputsize], [2, 0])
    model.setSolution(l if isinstance(l, list) else [l])

# returns the embedding for a Word (to be looked up in model.vocab)
def getVector(model, word):
    return model.solution[0][word.index]

# saves the embeddings from a trained solution in a model to file
def save(fname, model, binary=False):
    s = sorted(model.vocab.items(), key=lambda x: x[1].index)
    if binary:
        print("write binary")
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

# loads the embeddings saved to file
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
