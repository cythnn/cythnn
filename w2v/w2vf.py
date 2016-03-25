import cython
from w2v import *
from numpy import int32
import numpy as np
import cProfile
from skipgram import sg_py, model

if __name__ == "__main__":
    #sentences = "aap noot mies wim zus jet aap noot mies aap aap".split(" ")
    #text8 can be downloaded from http://mattmahoney.net/dc/text8.zip
    sentences = open("text8").read().split(" ")
    print("read %d" % (len(sentences)))

    vocab = build_vocab(sentences)
    solution = Solution(sentences)
    print("start lookup")
    sen = vocab.lookup_wordids(sentences)
    print("lookup up")
    #print(" ".join([ w + str(x) for w, x in zip(sentences, sen) ]))

    m = model(vocab, solution, 1, 0.025, 5)
    print("model created")
    sg_py(0, m, sen)
    print("model trained")
    # model.print()
    print (vocab.lookup_vector("king", solution))

    #print("king %s" % vocab['king'].syn0)
    #print("queen %s" % (model.get('queen')))
    print(vocab.similarity('king', 'king', solution))
    print(vocab.similarity('king', 'queen', solution))
    print(vocab.similarity('man', 'woman', solution))
    print(vocab.similarity('king', 'man', solution))
    print(vocab.similarity('queen', 'woman', solution))
    print(vocab.similarity('queen', 'anarchy', solution))

    r = most_similar('king', 'woman', 'man', vocab, solution, 10)

    save("vecf.w", vocab, solution)

    print(r)

    print("done")

