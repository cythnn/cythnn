from arch.Word2Vec import Word2Vec
from model.model import Model
from pipe.ContextWindows import contextWindow
from pipe.ConvertWordIds import convertWordIds
from pipe.DownSample import DownSample
from tools.word2vec import save
from tools.worddict import buildvocab
from pipe.createInputTasks import createW2VInputTasks

# Word2Vec uses Skipgram with Hierarchical Softmax by default
# setting cacheinner > 0 uses the caching variant that allows for faster parallel learning using multiple cores
# updatecacherate determines how often the cache is written back, e.g. 1 means after every trained word
# which in our experiments is often close to optimal, higher settings are more efficient when using
# a smaller windowsize or using more cores than optimal
if __name__ == "__main__":
    m = Model(alpha=0.025, vectorsize=100,
                 input="data/text8",
                 inputrange=None, # means all
                 build=[ buildvocab ],
                 pipeline=[ createW2VInputTasks, convertWordIds, DownSample, contextWindow, Word2Vec ],
                 mintf=5, cores=2, threads=3, windowsize=5, iterations=1, downsample=0.001,
                 updatecacherate=1, cacheinner=31, # cache the 31 most frequently appearing nodes in the Huffmann tree
            )
    m.run()
    save("results/vectors.sghs.bin", m, binary=True)

