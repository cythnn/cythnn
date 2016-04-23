from arch.SkipgramHS import SkipgramHS
from convertWordIds.py import convertWordIds
from model.model import Model
from tools.taketime import taketime
from tools.word2vec import createW2V, save
from tools.worddict import build_vocab
from w2vContextWindows.cy import contextWindow
from w2vHSoftmax.cy import build_hs_tree
from datetime import datetime

def doTestSkipgramHS(inputrange=None):
    return Model(alpha=0.025, vectorsize=100,
                 input="data/text8",
                 inputrange=inputrange,
                 build=[ build_vocab, build_hs_tree, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, SkipgramHS ],
                 mintf=5, cores=2, threads=3, windowsize=5, iterations=1, split=1)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = doTestSkipgramHS()
    #m = doTestSkipgramHS(inputrange=range(10000))
    time(m)
    save("results/vectors.sghs.bin", m, binary=True)
    print("done", str(datetime.now()))

