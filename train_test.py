from convertWordIds.py import convertWordIds
from w2vContextWindows.cy import contextWindow
from w2vSkipgramHS.cy import trainSkipgramHS
from model.cy import model
from tools.nntools import createW2V, save
from tools.worddict import build_vocab
from tools.wordio import wordStreams
from w2vHSoftmax.cy import build_hs_tree

def doTrain():
    return model(alpha = 0.025, vectorsize=100,
                 input = wordStreams("data/test", byterange=None, parts=4),
                 build=[ build_vocab, build_hs_tree, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, trainSkipgramHS ],
                 mintf=1, threads=1, windowsize=1, iterations=1)

if __name__ == "__main__":
    m = doTrain() #tiny example to experiment on

    m.run()

    save("results/vectors.test", m)
    save("results/vectors.test.bin", m, binary=True)

    print("done")

