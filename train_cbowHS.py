from convertWordIds.py import convertWordIds
from tools.taketime import taketime
from w2vContextWindows.cy import contextWindow
from w2vCbowHS.cy import trainCbowHS
from model.cy import model
from tools.nntools import createW2V, save
from tools.worddict import build_vocab
from tools.wordio import wordStreams
from w2vHSoftmax.cy import build_hs_tree

def doTestCbowHS(inputrange=None):
    return model(alpha=0.05, vectorsize=100,
                 input="data/text8",
                 inputrange=inputrange,
                 build=[ build_vocab, build_hs_tree, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, trainCbowHS ],
                 mintf=5, threads=2, windowsize=5, iterations=1)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = doTestCbowHS()

    time(m)

    save("results/vectors.cbhs", m)
    save("results/vectors.cbhs.bin", m, binary=True)

    print("done")

