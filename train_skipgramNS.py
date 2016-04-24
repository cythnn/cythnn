from arch.SkipgramNS import SkipgramNS
from pipe.ConvertWordIds import convertWordIds
from model.model import Model
from tools.taketime import taketime
from tools.word2vec import save
from tools.worddict import build_vocab
from pipe.ContextWindow import contextWindow


def doTestSkipgramNS(inputrange=None):
    return Model(alpha=0.025, vectorsize=100,
                 input="data/text8",
                 inputrange=inputrange,
                 build=[ build_vocab ],
                 pipeline=[ convertWordIds, contextWindow, SkipgramNS ],
                 mintf=5, cores=2, threads=3, windowsize=5, iterations=1, negative=5)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = doTestSkipgramNS()
    time(m)
    save("results/vectors.sgns.bin", m, binary=True)
    print("done")

