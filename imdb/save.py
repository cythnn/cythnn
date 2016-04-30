# saves the embeddings from a trained solution in a model to file
from tools.matrix import createMatrices
import numpy as np
from numpy import float32

def save(fname, model):
    s = sorted(model.vocab.items(), key=lambda x: x[1].index)
    solution = model.getSolution()
    with open(fname, 'w') as fout:
        fout.write("%s %s\n" % (len(model.itemids), solution.getLayerSize(1)))
        for index, imdbid in enumerate(model.itemids):
            row = model.matrices[0][index]
            fout.write("%s %s\n" % (imdbid, ' '.join("%f" % val for val in row)))

def load(fname):
    items = dict()
    with open(fname, 'r') as fin:
        itemcount, vectorsize = fin.readline().split(" ")
        items = dict(itemcount)
        embeddings = np.zeros((itemcount, vectorsize), dtype=float32)
        index = 0
        for line in fin.readlines():
            terms = line.split(' ')
            imdbid = terms[0]
            for i in range(vectorsize):
                embeddings[index, i] = terms[i + 1]
            items[imdbid] = embeddings[index]
            index += 1
        return embeddings

