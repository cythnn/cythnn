from nqe.findmax import *

if __name__ == "__main__":
    vectorfile = sys.argv[1]
    wordfile = sys.argv[2]
    table, word2vec, vec2word = loadvectors(vectorfile)
    #words = loadAnalogy(wordfile, word2vec)
    words = [ 1, 2 ]
    a = findmax(words, table, table.shape()[1], table.shape()[0])
    print(a)


