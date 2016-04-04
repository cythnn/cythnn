from skipgram.skipgram import sg_py
from tools.nntools import Solution, createW2V, save
from tools.worddict import build_vocab
from tools.wordio import wordStreams
from tools.nnmodel.model import model

def doTrain():
    wordstreams = wordStreams("../test", byterange=None, parts=1)
    print([x for x in wordstreams[0]])
    wordstreams = wordStreams("../test", byterange=None, parts=1)
    vocab = build_vocab(wordstreams, MIN_TF=1)
    solution = createW2V(vocab, 2)
    wordstreams = wordStreams("../test", byterange=None, parts=1)
    target = vocab.get('aap').index
    m = model(vocab, solution, 1, 0.025, windowsize=1, iterations=1, target=-2)
    return wordstreams, vocab, solution, m

def doTest(byterange=None):
    wordstreams = wordStreams("../text8", byterange=byterange, parts=2)
    vocab = build_vocab(wordstreams, MIN_TF=5)
    solution = createW2V(vocab, 100)
    wordstreams = wordStreams("../text8", byterange=byterange, parts=1)
    #target = vocab.get('king').index
    m = model(vocab, solution, 1, 0.025, windowsize=5, iterations=1, target=-1)
    return wordstreams, vocab, solution, m

if __name__ == "__main__":
    #sentences = "aap noot mies wim zus jet aap noot mies aap aap".split(" ")
    #text8 can be downloaded from http://mattmahoney.net/dc/text8.zip

    wordstreams, vocab, solution, model = doTrain()
    #wordstreams, vocab, solution, model = doTest(byterange=range(1000000))  #for w, word in vocab.items():
    #wordstreams, vocab, solution, model = doTest()

    print("start lookup")
    sen = vocab.lookup_wordids(wordstreams[0])

    #print(sen[:1000])
    print("lookup up")

    print("model created")
    sg_py(0, model, sen)
    print("model trained")
    # model.print()
    #runTests(vocab, solution)

    save("tt.w", vocab, solution)
    save("ttb.w", vocab, solution, binary=True)

    print("done")

