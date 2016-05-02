from pipe.ConvertWordIds import convertWordIds
from pipe.DownSample import DownSample
from pipe.createInputTasks import createW2VInputTasks
from tools.taketime import taketime
from pipe.ContextWindows import contextWindow
from arch.CbowNS import CbowNS
from model.model import Model
from tools.word2vec import save
from tools.worddict import buildvocab

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = Model(alpha=0.05, vectorsize=100,
                 input="data/text8",
                 inputrange=None, # means all
                 build=[ buildvocab ],
                 pipeline=[ createW2VInputTasks, convertWordIds, DownSample, contextWindow, CbowNS ],
                 mintf=5, cores=2, threads=3, windowsize=5, downsample=0, iterations=1, negative=5)
    time(m)
    save("results/vectors.cbns.bin", m, binary=True)

