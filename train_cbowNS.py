from convertWordIds.py import convertWordIds
from tools.taketime import taketime
from w2vContextWindows.cy import contextWindow
from w2vCbowNS.cy import trainCbowNS
from model.cy import model
from tools.nntools import createW2V, save
from tools.worddict import build_vocab
from tools.wordio import wordStreams

def doTestCbowNS(inputrange=None):
    return model(alpha=0.05, vectorsize=100,
                 input="data/text8",
                 inputrange=inputrange,
                 build=[ build_vocab, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, trainCbowNS ],
                 mintf=5, cores=2, windowsize=5, iterations=1, negative=5)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = doTestCbowNS()
    time(m)
    save("results/vectors.cbns.bin", m, binary=True)

    print("done")

