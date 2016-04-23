from convertWordIds.py import convertWordIds
from tools.taketime import taketime
from w2vContextWindows.cy import contextWindow
from arch.CbowHS import CbowHS
from model.model import Model
from tools.word2vec import createW2V, save
from tools.worddict import build_vocab
from w2vHSoftmax.cy import build_hs_tree

def doTestCbowHS(inputrange=None):
    return Model(alpha=0.05, vectorsize=100,
                 input="data/text8",
                 inputrange=inputrange,
                 build=[ build_vocab, build_hs_tree, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, CbowHS ],
                 mintf=5, cores=2, threads=3, windowsize=5, iterations=1, split=1)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    #m = doTestCbowHS(inputrange=range(1000000))
    m = doTestCbowHS()
    time(m)
    save("results/vectors.cbhs1.bin", m, binary=True)

    print("done")

