from tools.nntools import Solution, createW2V, save
from tools.worddict import build_vocab
from tools.wordio import wordStreams
from tools.nnmodel.model import model
from w2v.train.stream import stream
import numpy

from w2v.train.train import train

def doTrain():
    wordstreams = wordStreams("../../test", byterange=None, parts=1)
    print([x for x in wordstreams[0]])
    wordstreams = wordStreams("../../test", byterange=None, parts=1)
    vocab = build_vocab(wordstreams, MIN_TF=1)
    solution = createW2V(vocab, 2)
    wordstreams = wordStreams("../../test", byterange=None, parts=1)
    m = model(vocab, solution, 1, 0.025, windowsize=1, iterations=1, target=-2)
    print([w for w in wordstreams[0]])
    return wordstreams, vocab, solution, m

def doTest(byterange=None):
    wordstreams = wordStreams("../../text8", byterange=byterange, parts=2)
    vocab = build_vocab(wordstreams, MIN_TF=5)
    solution = createW2V(vocab, 100)
    wordstreams = wordStreams("../../text8", byterange=byterange, parts=4)
    #target = vocab.get('king').index
    m = model(vocab, solution, 1, 0.025, windowsize=5, iterations=1, target=-1)
    return wordstreams, vocab, solution, m


if __name__ == "__main__":
    #wordstreams, vocab, solution, model = doTrain()  #for w, word in vocab.items():
    #wordstreams, vocab, solution, model = doTest( byterange=range(1000000))
    wordstreams, vocab, solution, model = doTest()

    for i, s in enumerate(wordstreams):
        print("part", i)
        str = stream(vocab, s, model.windowsize)
        samples = numpy.fromiter(str, dtype='i4,i4,i4')
        train(0, model, samples, model.start_alpha)

    # model.print()
    #runTests(vocab, solution)

    save("tt.w", vocab, solution)
    save("ttb.w", vocab, solution, binary=True)

    print("done")

