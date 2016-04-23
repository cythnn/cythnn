import os, math, re
import numpy as np
import struct

from numpy import int32, uint64, float32

from matrix.cy import createMatrices
from tools.worddict import Vocabulary, Word

# creates a solution space for a word2vec model, 2 weight matrices, initialized resp. random and with zeros
#@taketime("createW2V")
def createW2V(learner, model):
    if isinstance(model.vocab, Vocabulary):
        l = createMatrices([len(model.vocab), model.vectorsize, model.outputsize], [2, 0])
    else:
        l = createMatrices([model.vocab, model.vectorsize, model.outputsize], [2, 0])
    model.setSolution(l if isinstance(l, list) else [l])

# returns the embedding for a Word (to be looked up in model.vocab)
def getVector(model, word):
    return model.matrices[0][word.index]

# saves the embeddings from a trained solution in a model to file
def save(fname, model, binary=False):
    s = sorted(model.vocab.items(), key=lambda x: x[1].index)
    solution = model.getSolution()
    if binary:
        print("write binary")
        with open(fname, 'wb') as fout:
            fout.write(("%s %s\n" % (len(model.vocab), solution.getLayerSize(1))).encode())
            for word, obj in s:
                row = getVector(model, obj)
                fout.write((word + " ").encode())
                fout.write(struct.pack('%sf' % len(row), *row))
    else:
        print("write flat")
        with open(fname, 'w') as fout:
            fout.write("%s %s\n" % (len(model.vocab), solution.getLayerSize(1)))
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

# normalize a vector (commonly used before comparing vectors)
def normalize(w1):
   return (w1 / math.sqrt(sum([ w * w for w in w1 ])))

