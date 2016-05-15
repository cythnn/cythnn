from arch.Word2Vec import Word2Vec
from pipe.ConvertWordIds import convertWordIds
from pipe.DownSample import DownSample
from pipe.createInputTasks import createW2VInputTasks
from pipe.ContextWindows import contextWindow
from model.model import Model
from tools.word2vec import save
from tools.worddict import buildvocab

# set cbow=1 to use CBOW, Hierarchical Softmax is used by default if negative is not set (higher than 0)
if __name__ == "__main__":
    m = Model(alpha=0.05, vectorsize=100,
                 input="data/text8",
                 inputrange=None, # means all
                 build=[ buildvocab ],
                 pipeline=[ createW2VInputTasks, convertWordIds, DownSample, contextWindow, Word2Vec ],
                 mintf=5, cores=2, threads=3, windowsize=5, iterations=1, downsample=0.001,
                 cbow=1,
              )
    m.run()
    save("results/vectors.cbhs.bin", m, binary=True)

