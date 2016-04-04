from multiprocessing.pool import Pool

from oldtools.worddict import wordStreams, countWords, mergeDicts

import numpy as np
from numpy import float32, empty, random, uint32, uint64, int32
import struct, math

from tools.taketime import taketime

MAX_EXP = 6
EXP_TABLE = 1000
ALPHA_START = 0.025

def normalize(w1):
   return (w1 / math.sqrt(sum([ w * w for w in w1 ])))

def countW(rnge, file):
    return countWords()

def build_vocab(wordstreams, MIN_TF = 5, cores = 2, parts = 2):
    pool = Pool(processes = cores)

    tokens = pool.map(countWords, wordstreams)
    merged = mergeDicts(tokens)
    return Vocabulary(merged, MIN_TF)

def exptable():
    #fout = open("log", "w")
    table = np.empty((EXP_TABLE), float32)
    for i in range(0, 1000):
        e = math.exp(float32(2 * MAX_EXP * i / EXP_TABLE - MAX_EXP))
        table[i] = e / float32(e + 1)
        #fout.write("%d %.10f\n"%(i, table[i]))
    #fout.close()
    return table

def create_binary_tree(vocab):
    newnodes = []
    pos1 = len(vocab) - 1
    pos2 = 0

    for a in range(len(vocab)-1):
        if pos1 >= 0:
            if pos2 >= len(newnodes) or vocab[pos1].count < newnodes[pos2].count:
                left = vocab[pos1]
                pos1 -= 1
            else:
                left = newnodes[pos2];
                pos2 += 1
        else:
            left = newnodes[pos2]
            pos2+=1
        if pos1 >= 0:
            if pos2 >= len(newnodes) or vocab[pos1].count < newnodes[pos2].count:
                right = vocab[pos1]
                pos1 -= 1
            else:
                right = newnodes[pos2]
                pos2 += 1
        else:
            right = newnodes[pos2]
            pos2 += 1
        newnode = Word(left.count + right.count, index=len(vocab) - a - 2)
        left.parent = newnode
        right.parent = newnode
        #print("merge %d -> %d %d"%(newnode.index, left.index, right.index))
        right.isRight = True
        newnodes.append(newnode)
    print("tree build")

    for word in vocab:
        w = word.parent
        path_length, leftright, nodes = 0, [word.isRight], []
        while hasattr(w, 'index') and w.index > 0:
            nodes.append(w)
            leftright.append(w.isRight)
            w = w.parent

        nodes.append(w)
        leftright.reverse()
        nodes.reverse()
        word.innernodes = [(l,n) for l, n in zip(leftright, nodes)]
    print("codes assigned")

def train(vocab, solution, words, alpha_start=ALPHA_START, window=5):
    vector_size = solution.vector_size
    expTable = exptable()
    next_random = uint64(1)
    words_processed = 0
    words_batch = 0
    alpha = alpha_start

    words = vocab.lookup_words(words)
    for sentence_position in range(0, len(words)):
        word = words[sentence_position]
        if words_processed >= 10000 * words_batch:
            words_batch += 1
            alpha = ALPHA_START * (1 - words_processed / (vocab.total_words + 1))
            if alpha < ALPHA_START * 0.0001:
                alpha = ALPHA_START * 0.0001
            print("alpha=%f %d" % (alpha, words_processed))

        with np.errstate(over='ignore'):
            next_random = next_random * uint64(25214903917) + uint64(11);
        b = int(next_random % window);
        for a in range(b, window * 2 + 1 - b):
            if a != window:
                c = sentence_position - window + a
                if (c >= 0 and c < len(words)):
                    target_word = words[c]
                    l1 = target_word.index
                    syn0 = solution.syn0[l1]
                    neu1e = np.zeros(vector_size)
                    for expy, inner in word.innernodes:
                        l2 = inner.index
                        syn1 = solution.syn1[l2]
                        e = np.dot(syn0, syn1)
                        if e > -MAX_EXP and e < MAX_EXP:
                            y = expTable[(e + MAX_EXP) * (EXP_TABLE / MAX_EXP / 2)]
                            g = (1 - expy - y) * alpha
                            neu1e += g * syn1
                            syn1 += g * syn0
                    syn0 += neu1e
        words_processed += 1

def save(fname, vocab, solution, binary=False):
    s = sorted(vocab.items(), key=lambda x: x[1].index)
    if binary:
        with open(fname, 'wb') as fout:
            fout.write(("%s %s\n" % (len(vocab), solution.vector_size)).encode())
            for word, obj in s:
                row = obj.getVector(solution)
                fout.write((word + " ").encode())
                fout.write(struct.pack('%sf' % len(row), *row))
    else:
        print("write flat")
        with open(fname, 'w') as fout:
            fout.write("%s %s\n" % (len(vocab), solution.vector_size))
            for word, obj in s:
                row = obj.getVector(solution)
                fout.write("%s %d %s\n" % (word, obj.count, ' '.join("%f" % val for val in row)))


def load(fname, binary=False, normalized=False):
    with open(fname, 'r') as fin:
        header = fin.readline()
        wordcount, vector_size = [ int(x) for x in header.split(" ") ]
        solution = Solution(vocabularysize=wordcount, vector_size=vector_size)
        vocab = Vocabulary({}, MIN_TF=-1)
        index = 0
        for line in fin.readlines():
            terms = line.split(" ")
            word = terms[0]
            count = int(terms[1])
            solution.syn0[index] = [float32(terms[i]) for i in range(2, len(terms))]
            if normalized:
                solution.syn0[index] = normalize(solution.syn0[index])
            vocab[word] = Word(count, index=index, word=word)
            index+=1
            vocab.total_words += count
        return vocab, solution

class Vocabulary(dict):
    def __init__(self, vocab, MIN_TF):
        super(Vocabulary, self).__init__()

        vocab = sorted(vocab.items(), key=lambda x: -x[1])

        words = [Word(0, word="</s>", index=0)]
        for word, count in vocab:
            if count >= MIN_TF and word != "</s>":
                words.append(Word(count, word=word, index=len(words)))
        create_binary_tree(words)

        total_words = 0
        for word in words:
            self.__setitem__(word.word, word)
            total_words += word.count
        self.total_words = total_words
        print("Vocabulary %d %d"%(len(self), total_words))

    def lookup_words(self, sentence):
        result = []
        for word in sentence:
            dict_word = self.get(word)
            if (dict_word != None):
                result.append(dict_word)
        return result

    @taketime("lookup_wordids")
    def lookup_wordids(self, sentence):
        result = []
        for word in sentence:
            dict_word = self.get(word)
            if (dict_word is not None):
                result.append(dict_word.index)
                if len(result) < 100:
                    print("%s %d %s"%(word, dict_word.index, dict_word.word))
        result = np.array(result, dtype=int32 )
        #print(result)
        return result

    def lookup_vector(self, term, solution):
        word = self.get(term)
        return solution.syn0[word.index]

    def similarity(self, term, term2, solution):
        return np.dot(self.get(term).getNormalized(solution), self.get(term2).getNormalized(solution))

def most_similar(a, b, c, vocab, solution, topn=10):
    w = np.subtract(vocab.get(b).getNormalized(solution), vocab.get(a).getNormalized(solution))
    w = np.add(w, vocab.get(c).getNormalized(solution))
    w = normalize(w)
    sim = [(np.dot(x.getNormalized(solution), w), x) for xt, x in vocab.items()]
    sim.sort(key=lambda t: t[0], reverse=True)
    r = [(score, x.word) for (score,x) in sim][:topn]
    return r

class Solution:
    def __init__(self, vocab = None, vocabularysize = None, vector_size = 100):
        self.vector_size = vector_size
        if (vocabularysize != None):
            self.syn0 = np.ndarray((vocabularysize, vector_size))
        else:
            if vocab != None:
                next_random = uint64(1)
                self.syn0 = np.empty((len(vocab), vector_size), dtype=float32)
                for a in range(len(vocab)):
                    for b in range(vector_size):
                        with np.errstate(over='ignore'):
                            next_random = next_random * uint64(25214903917) + uint64(11);
                        self.syn0[a, b] = ((next_random & uint64(65535)) / float32(65536) - 0.5) / vector_size
                #self.syn0 = np.random.uniform(-0.5 / vector_size, 0.5 / vector_size, (len(vocab), vector_size))
                self.syn1 = np.zeros((len(vocab)-1, vector_size), dtype=float32)

class Word:
    next_random = uint64(1)

    def __init__(self, count, index, word=None):
        self.count = count
        self.isRight = False
        self.parent = None
        if word != None:
            self.word = word
        self.index = index

    def getNormalized(self, solution):
        if not hasattr(self, 'normalized'):
            self.normalized = normalize(solution.syn0[self.index])
        return self.normalized

    def getVector(self, solution):
        return solution.syn0[self.index]

    def __str__(self):
        if hasattr(self, 'word'):
            return self.word
        else:
            return str(self.index)

def runTests(vocab, solution):
    print(vocab.lookup_vector("king", solution))

    print(vocab.similarity('king', 'king', solution))
    print(vocab.similarity('king', 'queen', solution))
    print(vocab.similarity('man', 'woman', solution))
    print(vocab.similarity('king', 'man', solution))
    print(vocab.similarity('queen', 'woman', solution))
    print(vocab.similarity('queen', 'anarchy', solution))

    print(most_similar('king', 'queen', 'man', vocab, solution, 10))
    print(most_similar('queen', 'woman', 'king', vocab, solution, 10))
    print(most_similar('man', 'woman', 'king', vocab, solution, 10))
    print(most_similar('woman', 'man', 'queen', vocab, solution, 10))

if __name__ == "__main__":
    wordstreams = wordStreams("../test", byterange = None, parts=1)
    vocab = build_vocab(wordstreams)
    solution = Solution(vocab, vector_size=2)
    wordstreams = wordStreams("../test", byterange = None, parts=1)
    train(vocab, solution, wordstreams[0], window=1)
    # runTests(vocab, solution)

    save("test.w", vocab, solution)

    print("done")

