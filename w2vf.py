from functools import partial

from hs.hs import processhs, build_hierarchical_softmax
from model.model import model
from tools.nntools import createW2V, save
from tools.worddict import build_vocab, readWordIds
from tools.wordio import wordStreams
from w2vTrainer.train import addTrainW2V

def doTrain():
    return model(alpha = 0.025,
                 pybuild=[build_vocab, build_hierarchical_softmax, createW2V ],
                 pipeline=[readWordIds, processhs],
                 pipelinec=[addTrainW2V],
                 input = wordStreams("data/test", byterange=None, parts=2),
                 vectorsize=100, mintf=1, cores=2, windowsize=1, iterations=1, debugtarget=None)

def doTest(byterange=None):
    return model(alpha=0.025, vectorsize=100,
                 pybuild=[build_vocab, createW2V, build_hierarchical_softmax],
                 pipeline=[readWordIds, processhs],
                 pipelinec=[addTrainW2V],
                 input=wordStreams("data/text8", byterange=byterange, parts=2),
                 mintf=5, cores=2, windowsize=5, iterations=1, debugtarget=None)

def doTest2(byterange=None):
    return model(alpha=0.025,
                 vectorsize=100,
                 input=wordStreams("data/text8", byterange=byterange, parts=8),
                 pybuild=[build_vocab, createW2V, build_hierarchical_softmax],
                 pipeline=[readWordIds, processhs],
                 pipelinec=[addTrainW2V],
                 mintf=5, cores=2, windowsize=5, iterations=1, debugtarget=None)

if __name__ == "__main__":
    model = doTest()  #for w, word in vocab.items():
    #model = doTest( byterange=range(1000000) )
    #model = doTrain()ZZ
    print(len(model.input), model.input[0])

    model.run()

    save("results/tt3.w", model)
    save("results/tt3b.w", model, binary=True)

    print("done")

