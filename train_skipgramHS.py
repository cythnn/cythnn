from arch.SkipgramHS import SkipgramHS
from pipe.Split import Split
from pipe.ConvertWordIds import convertWordIds
from model.model import Model
from pipe.DownSample import DownSample
from tools.taketime import taketime
from tools.word2vec import save
from tools.worddict import build_vocab
from pipe.ContextWindows import contextWindow
from datetime import datetime

def doTestSkipgramHS(inputrange=None):
    return Model(alpha=0.025, vectorsize=100,
                 input="data/text8",
                 inputrange=inputrange,
                 build=[ build_vocab ],
                 pipeline=[ convertWordIds, DownSample, contextWindow, Split, SkipgramHS ],
                 mintf=5, cores=2, threads=3, windowsize=5, iterations=1, split=1, sample=0.001)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = doTestSkipgramHS()
    #m = doTestSkipgramHS(inputrange=range(10000))
    time(m)
    save("results/vectors.sghs.bin", m, binary=True)
    print("done", str(datetime.now()))

