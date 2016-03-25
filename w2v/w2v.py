import heapq

import numpy as np
from numpy import float32, empty, random, uint32, uint64, int32
from scipy.spatial import distance
import operator, struct, math

MAX_EXP = 6
EXP_TABLE = 1000
ALPHA_START = 0.025

def normalize(w1):
   return (w1 / math.sqrt(sum([ w * w for w in w1 ])))

def scan_vocab(sentences):
    vocab = {}
    for word in sentences:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
    print("counted %d" % len(vocab))
    return Vocabulary(vocab)

def exptable():
    table = np.empty((EXP_TABLE), float32)
    for i in range(0, 1000):
        e = np.exp((i / EXP_TABLE * 2 - 1) * MAX_EXP)
        table[i] = e / (e + 1)
    return table

def create_binary_tree(vocab):
    heap = list(vocab.values())
    heapq.heapify(heap)
    while len(heap) > 1:
        left, right = heapq.heappop(heap), heapq.heappop(heap)
        right.isRight = True
        newnode = Word(left.count + right.count, index=len(heap))
        left.parent = newnode
        right.parent = newnode
        heapq.heappush(heap, newnode)
    print("tree build")

    for word in vocab.values():
        w = word.parent
        path_length, leftright, nodes = 0, [word.isRight], []
        while w.index > 0:
            nodes.append(w)
            leftright.append(w.isRight)
            w = w.parent

        nodes.append(w)
        leftright.reverse()
        nodes.reverse()
        word.innernodes = [(l,n) for l, n in zip(leftright, nodes)]
    print("codes assigned")

def build_vocab(sentences):
    vocab = scan_vocab(sentences)
    create_binary_tree(vocab)
    return vocab

def train(vocab, solution, sentences, alpha_start=ALPHA_START, vector_size=100, window=5):
    expTable = exptable()
    next_random = uint64(1)
    words_processed = 0
    words_batch = 0
    alpha = alpha_start

    words = vocab.lookup_words(sentences)
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
        b = next_random.item() % window;
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
    s = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    if binary:
        with open(fname, 'wb') as fout:
            fout.write(("%s %s\n" % (len(vocab), solution.vector_size)).encode())
            for word, obj in s:
                row = obj.getVector(solution)
                fout.write((word + " ").encode())
                fout.write(struct.pack('%sf' % len(row), *row))
    else:
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
        vocab = Vocabulary({})
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
    def __init__(self, vocab):
        super(Vocabulary, self).__init__()
        total_words = 0
        index = 0
        vocab = sorted(vocab.items(), key=lambda x: -x[1])
        for word, count in vocab:
            if (count >= 5):
                self.__setitem__(word, Word(count, word=word, index=index))
                index += 1
                total_words += count
        self.total_words = total_words
        print("Vocabulary %d %d"%(len(self), total_words))

    def lookup_words(self, sentence):
        result = []
        for word in sentence:
            dict_word = self.get(word)
            if (dict_word != None):
                result.append(dict_word.index)
        return result

    def lookup_wordids(self, sentence):
        result = np.ndarray((len(sentence)), dtype=int32)
        idx = 0
        for word in sentence:
            dict_word = self.get(word)
            if (dict_word != None):
                result[idx] = dict_word.index
                idx += 1
        result.resize((idx))
        return result

    def lookup_vector(self, term, solution):
        word = self.get(term)
        return solution.syn0[word.index]

    def similarity(self, term, term2, solution):
        return 1 - distance.cosine(self.get(term).getVector(solution),
                               self.get(term2).getVector(solution))

def most_similar(a, b, c, vocab, solution, topn=10):
    w = np.add(vocab.lookup_vector(b, solution),
               vocab.lookup_vector(c, solution))
    w = np.subtract(w, vocab.lookup_vector(a, solution))
    #w = normalize(w)
    dists = [(1 - distance.cosine(x.getVector(solution), w), x) for xt, x in vocab.items()]
    dists.sort(key=lambda t: t[0], reverse=True)
    r = [(score, x.word) for (score,x) in dists][:topn]
    return r

class Solution:
    def __init__(self, vocab = None, vocabularysize = None, vector_size = 100):
        self.vector_size = vector_size
        if (vocabularysize != None):
            self.syn0 = np.ndarray((vocabularysize, vector_size))
        else:
            if vocab != None:
                #self.syn0 = np.empty((len(vocab), vector_size), dtype=float32)
                self.syn0 = np.random.uniform(-0.5 / vector_size, 0.5 / vector_size, (len(vocab), vector_size))
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
            self.normalized = normalized(solution.syn0[self.index])
        return self.normalized

    def getVector(self, solution):
        return solution.syn0[self.index]

    def __lt__(self, other):
        # print("%d %d %s"%(other.count, self.count, other.count > self.count))
        return other.count > self.count

    def __str__(self):
        if hasattr(self, 'word'):
            return self.word
        else:
            return str(self.index)

if __name__ == "__main__":
    #sentences = "aap noot mies wim zus jet aap noot mies aap aap".split(" ")
    sentences = open("text8").read().split(" ")[:100000]
    print("read %d" % (len(sentences)))
    vocab = build_vocab(sentences)
    solution = Solution(sentences)
    train(vocab, solution, sentences)
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

    save("vec.w", vocab, solution, 100)

    print(r)

    print("done")

