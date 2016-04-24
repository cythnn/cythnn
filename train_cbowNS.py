from pipe.ConvertWordIds import convertWordIds
from tools.taketime import taketime
from pipe.ContextWindow import contextWindow
from arch.CbowNS import CbowNS
from model.model import Model
from tools.word2vec import save
from tools.worddict import build_vocab

def doTestCbowNS(inputrange=None):
    return Model(alpha=0.05, vectorsize=100,
                 input="data/text8",
                 inputrange=inputrange,
                 build=[ build_vocab ],
                 pipeline=[ convertWordIds, contextWindow, CbowNS ],
                 mintf=5, cores=2, threads=3, windowsize=5, iterations=1, negative=5)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = doTestCbowNS()
    time(m)
    save("results/vectors.cbns.bin", m, binary=True)

    print("done")

