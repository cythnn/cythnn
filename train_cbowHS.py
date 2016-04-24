from pipe.ConvertWordIds import convertWordIds
from tools.taketime import taketime
from pipe.ContextWindow import contextWindow
from arch.CbowHS import CbowHS
from model.model import Model
from tools.word2vec import save
from tools.worddict import build_vocab

def doTestCbowHS(inputrange=None):
    return Model(alpha=0.05, vectorsize=100,
                 input="data/text8",
                 inputrange=inputrange,
                 build=[ build_vocab ],
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

