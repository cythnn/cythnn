from convertWordIds.py import convertWordIds
from tools.taketime import taketime
from w2vContextWindows.cy import contextWindow
from w2vSkipgramHS.cy import trainSkipgramHS
from w2vSkipgramNS.cy import trainSkipgramNS
from w2vCbowHS.cy import trainCbowHS
from model.cy import model
from tools.nntools import createW2V, save
from tools.worddict import build_vocab
from tools.wordio import wordStreams
from w2vHSoftmax.cy import build_hs_tree

def doTrain():
    return model(alpha = 0.025, vectorsize=100,
                 input = wordStreams("data/test", byterange=None, parts=4),
                 build=[ build_vocab, build_hs_tree, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, trainSkipgramHS ],
                 mintf=1, cores=2, windowsize=1, iterations=1)

def doTestSkipgramHS(byterange=None):
    return model(alpha=0.025, vectorsize=100,
                 input=wordStreams("data/text8", byterange=byterange, parts=2),
                 build=[ build_vocab, build_hs_tree, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, trainSkipgramHS ],
                 mintf=5, cores=2, windowsize=5, iterations=1)

def doTestSkipgramNS(byterange=None):
    return model(alpha=0.025, vectorsize=100,
                 input=wordStreams("data/text8", byterange=byterange, parts=2),
                 build=[ build_vocab, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, trainSkipgramNS ],
                 mintf=5, cores=2, windowsize=5, iterations=1, negative=5)

def doTestCbowHS(byterange=None):
    return model(alpha=0.05, vectorsize=100,
                 input=wordStreams("data/text8", byterange=byterange, parts=2),
                 build=[ build_vocab, build_hs_tree, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, trainCbowHS ],
                 mintf=5, cores=2, windowsize=5, iterations=1)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    #m = doTestSkipgramHS()  #for w, word in vocab.items():
    #m = doTestSkipgramHS( byterange=range(1000000) ) # specify byterange to truncate the input
    #m = doTestCbowHS()  #for w, word in vocab.items():
    m = doTestSkipgramNS()  #for w, word in vocab.items():
    #m = doTrain() #tiny example to experiment on

    time(m)

    save("results/tt3.w", m)
    save("results/tt3b.w", m, binary=True)

    print("done")

