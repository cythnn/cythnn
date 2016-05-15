from arch.Word2Vec import Word2Vec
from pipe.ConvertWordIds import convertWordIds
from model.model import Model
from pipe.DownSample import DownSample
from pipe.createInputTasks import createW2VInputTasks
from tools.word2vec import save
from tools.worddict import buildvocab
from pipe.ContextWindows import contextWindow

# Word2Vec uses Skipgram by default, set negative > 0 to use negative sampling instead of Hierarchical Softmax
if __name__ == "__main__":
    m = Model(alpha=0.025, vectorsize=100,
                 input="data/text8",
                 inputrange=None, # means all
                 build=[ buildvocab ],
                 pipeline=[ createW2VInputTasks, convertWordIds, DownSample, contextWindow, Word2Vec ],
                 mintf=5, cores=2, threads=3, windowsize=5, downsample=0.001, iterations=1, negative=5)
    m.run()
    save("results/vectors.sgns.bin", m, binary=True)

