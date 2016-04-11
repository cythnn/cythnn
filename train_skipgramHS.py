from convertWordIds.py import convertWordIds
from tools.taketime import taketime
from w2vContextWindows.cy import contextWindow
from w2vSkipgramHS.cy import trainSkipgramHS
from model.cy import model
from tools.nntools import createW2V, save
from tools.worddict import build_vocab
from tools.wordio import wordStreams
from w2vHSoftmax.cy import build_hs_tree

def doTestSkipgramHS(byterange=None):
    return model(alpha=0.025, vectorsize=100,
                 input=wordStreams("data/text8", byterange=byterange, parts=2),
                 build=[ build_vocab, build_hs_tree, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, trainSkipgramHS ],
                 mintf=5, cores=2, windowsize=5, iterations=5)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = doTestSkipgramHS()  #for w, word in vocab.items():

    time(m)

    save("results/vectors.sghsi5d0", m)
    save("results/vectors.sghsi5d0.bin", m, binary=True)

    print("done")

