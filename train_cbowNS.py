from arch.Word2Vec import Word2Vec
from pipe.ConvertWordIds import convertWordIds
from pipe.DownSample import DownSample
from pipe.createInputTasks import createW2VInputTasks
from pipe.ContextWindows import contextWindow
from model.model import Model
from tools.word2vec import save
from tools.worddict import buildvocab

# Set model to cbow=1 and negative > 0 to learn using CBOW and negative sampling
if __name__ == "__main__":
    m = Model(alpha=0.05, vectorsize=100,
                 input="data/text8",
                 inputrange=None, # means all
                 build=[ buildvocab ],
                 pipeline=[ createW2VInputTasks, convertWordIds, DownSample, contextWindow, Word2Vec ],
                 mintf=5, cores=2, threads=3, windowsize=5, downsample=0.001, iterations=1,
                 cbow=1, negative=5)
    m.run()
    save("results/vectors.cbns.bin", m, binary=True)

