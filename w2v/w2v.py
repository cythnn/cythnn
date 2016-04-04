from tools.nntools import createW2V, Solution
from tools.worddict import *

import numpy as np
from numpy import float32, empty, random, uint32, uint64, int32
import struct

from tools.wordio import wordStreams

ALPHA_START = 0.025

def train(vocab, solution, words, alpha_start=ALPHA_START, window=5):
    vector_size = solution.vector_size
    expTable = solution.expTable
    next_random = uint64(1)
    words_processed = 0
    words_batch = 0
    alpha = alpha_start

    words = vocab.lookup_words(words)
    for sentence_position in range(0, len(words)):
        word = words[sentence_position]
        if words_processed >= 10000 * words_batch:
            words_batch += 1
            alpha = ALPHA_START * (1 - words_processed / (vocab.total_words + 1))
            if alpha < ALPHA_START * 0.0001:
                alpha = ALPHA_START * 0.0001
            print("alpha=%f %d" % (alpha, words_processed))

        with np.errstate(over='ignore'):
            next_random = next_random * uint64(25214903917) + uint64(11);
        b = int(next_random % window);
        for a in range(b, window * 2 + 1 - b):
            if a != window:
                c = sentence_position - window + a
                if (c >= 0 and c < len(words)):
                    target_word = words[c]
                    l1 = target_word.index
                    syn0 = solution.syn0[l1]
                    neu1e = np.zeros(vector_size)
                    for expy, inner in word.innernodes:
                        l2 = inner.index
                        syn1 = solution.syn1[l2]
                        e = np.dot(syn0, syn1)
                        if e > -solution.MAX_EXP and e < solution.MAX_EXP:
                            y = expTable[(e + solution.MAX_EXP) * (solution.EXP_TABLE / solution.MAX_EXP / 2)]
                            g = (1 - expy - y) * alpha
                            neu1e += g * syn1
                            syn1 += g * syn0
                    syn0 += neu1e
        words_processed += 1



def most_similar(a, b, c, vocab, solution, topn=10):
    w = np.subtract(vocab.get(b).getNormalized(solution), vocab.get(a).getNormalized(solution))
    w = np.add(w, vocab.get(c).getNormalized(solution))
    w = normalize(w)
    sim = [(np.dot(x.getNormalized(solution), w), x) for xt, x in vocab.items()]
    sim.sort(key=lambda t: t[0], reverse=True)
    r = [(score, x.word) for (score,x) in sim][:topn]
    return r



def runTests(vocab, solution):
    print(vocab.lookup_vector("king", solution))

    print(vocab.similarity('king', 'king', solution))
    print(vocab.similarity('king', 'queen', solution))
    print(vocab.similarity('man', 'woman', solution))
    print(vocab.similarity('king', 'man', solution))
    print(vocab.similarity('queen', 'woman', solution))
    print(vocab.similarity('queen', 'anarchy', solution))

    print(most_similar('king', 'queen', 'man', vocab, solution, 10))
    print(most_similar('queen', 'woman', 'king', vocab, solution, 10))
    print(most_similar('man', 'woman', 'king', vocab, solution, 10))
    print(most_similar('woman', 'man', 'queen', vocab, solution, 10))

if __name__ == "__main__":
    wordstreams = wordStreams("../test", byterange = None, parts=1)
    vocab = build_vocab(wordstreams)
    solution = createW2V(vocab, vector_size=2)
    wordstreams = wordStreams("../test", byterange = None, parts=1)
    train(vocab, solution, wordstreams[0], window=1)
    # runTests(vocab, solution)

    save("test.w", vocab, solution)

    print("done")

