from w import *

if __name__ == "__main__":
    vocab, solution = load("vecn.w", normalized=False)
    print (vocab.lookup_vector("king", solution))

    # print("king %s" % vocab['king'].syn0)
    # print("queen %s" % (model.get('queen')))
    print(vocab.similarity('king', 'king', solution))
    print(vocab.similarity('king', 'queen', solution))
    print(vocab.similarity('man', 'woman', solution))
    print(vocab.similarity('king', 'man', solution))
    print(vocab.similarity('queen', 'woman', solution))
    print(vocab.similarity('queen', 'anarchy', solution))
    r = most_similar('king', 'queen', 'man', vocab, solution, 10)
    print(r)
    r = most_similar('queen', 'king', 'woman', vocab, solution, 10)
    print(r)
    r = most_similar('woman', 'man', 'queen', vocab, solution, 10)
    print(r)
    r = most_similar('man', 'woman', 'king', vocab, solution, 10)
    print(r)
