from tools.worddict import build_vocab
from tools.wordio import wordStreams
from train.stream import stream

if __name__ == "__main__":
    wordstreams = wordStreams("../test", byterange=None, parts=1, window=0)
    vocab = build_vocab(wordstreams, MIN_TF=1)

    wordstreams = wordStreams("../test", byterange=None, parts=2, window=6)
    for w in wordstreams:
        s = stream(vocab, w, 1)
        print([x for x in s])

