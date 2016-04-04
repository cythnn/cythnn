from oldtools.worddict import wordStreams
from w2vold.w2v import save
from w2vold import *
from w2vold.skipgram import sg_py, model
from w2vold.w2v import build_vocab, Solution, exptable


def doTrain():
    wordstreams = wordStreams("../test", byterange=None, parts=1)
    print([x for x in wordstreams[0]])
    wordstreams = wordStreams("../test", byterange=None, parts=1)
    vocab = build_vocab(wordstreams, MIN_TF=1)
    solution = Solution(vocab, vector_size=2)
    wordstreams = wordStreams("../test", byterange=None, parts=1)
    target = vocab.get('aap').index
    m = model(vocab, solution, exptable(), 1, 0.025, windowsize=5, iterations=1, target=-2)
    return wordstreams, vocab, solution, m

def doTest():
    wordstreams = wordStreams("../text8", byterange=range(1000000), parts=2)
    vocab = build_vocab(wordstreams, MIN_TF=5)
    solution = Solution(vocab, vector_size=100)
    wordstreams = wordStreams("../text8", byterange=range(1000000), parts=1)
    target = vocab.get("king").index
    m = model(vocab, solution, exptable(), 1, 0.025, windowsize=5, iterations=1, target=target)
    return wordstreams, vocab, solution, m

if __name__ == "__main__":
    #sentences = "aap noot mies wim zus jet aap noot mies aap aap".split(" ")
    #text8 can be downloaded from http://mattmahoney.net/dc/text8.zip

    wordstreams, vocab, solution, model = doTrain()
    #wordstreams, vocab, solution, model = doTest()

    #for w, word in vocab.items():
    #    print("%d\t%s"%(word.index, w))
    #    if word.index == 10:
    #        break
    print("start lookup")
    sen = vocab.lookup_wordids(wordstreams[0])

    #print(sen[:1000])
    print("lookup up")

    print("model created")
    sg_py(0, model, sen)
    print("model trained")
    # model.print()
    #runTests(vocab, solution)

    save("old.w", vocab, solution)
    save("oldb.w", vocab, solution, binary=1)

print("done")

