from arch.Word2Vec import Word2Vec
from model.model import Model
from pipe.ContextWindows import contextWindow
from pipe.ConvertWordIds import convertWordIds
from pipe.DownSample import DownSample
from tools.word2vec import save
from tools.worddict import buildvocab
from pipe.CreateInputTasks import createW2VInputTasks

# Word2Vec uses Skipgram HS by default
if __name__ == "__main__":
    m = Model(alpha=0.025, vectorsize=100,
              input="data/text8",
              inputrange=range(1000000), #None, # means all
              build=[ buildvocab ],
              pipeline=[ createW2VInputTasks, convertWordIds, DownSample, contextWindow, Word2Vec ],
              mintf=5, cores=2, threads=3, windowsize=5, iterations=5, downsample=0.001,
              )
    m.run()
    save("results/vectors.sghs.bin", m, binary=True)

