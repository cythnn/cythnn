from numpy import zeros, float32, int32, dtype, fromstring
from tools.types cimport *
from numpy cimport *
from tools.blas cimport saxpy, scopy, snrm2

def loadvectors(fname):
    with open(fname, "r") as fin:
        header = fin.readline()
        vocab_size, vector_size = map(int, header.split())
        word2vec = {}
        vec2word = {}
        table = zeros((vocab_size, vector_size), dtype=float32)

        def add_word(word, weights):
            word_id = len(word2vec)
            if word in word2vec:
                print("duplicate word '%s' in %s, ignoring all but first", word, fname)
                return
            table[word_id] = weights
            word2vec[word] = word_id
            vec2word[word_id] = word

        binary_len = dtype(float32).itemsize * vector_size
        #for line_no in range(vocab_size):
        for line_no in range(100):
            word = []
            while True:
                ch = fin.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':
                    word.append(ch)
            word = b''.join(word)
            weights = fromstring(fin.read(binary_len), dtype=float32)
            add_word(word, weights)
    return table, word2vec, vec2word

def loadAnalogy(fname, word2vec):
    lines = zeros(20000 * 2 * 2, dtype=int32)
    i = 0
    with open(fname, "r") as fin:
        for line in fin.readlines():
            if not line.startswith(': '):
                a, b, c, d = [word.lower() for word in line.split()]
                if a in word2vec:
                    if b in word2vec:
                        lines[i] = word2vec[a]
                        lines[i + 1] = word2vec[b]
                        i += 2
                    if c in word2vec:
                        lines[i] = word2vec[a]
                        lines[i + 1] = word2vec[c]
                        i += 2
    lines.resize(i)
    return lines

cdef float fmONE = -1.0
cdef int iONE = 1

cdef cREAL findmax(ndarray llines, ndarray ttable, int vectorsize, int vocsize):
    cdef int v, w, i, length = len(llines)
    cdef cREAL vlength
    cdef cREAL maxlength = 0
    cdef cINT *lines = toIArray(llines)
    cdef cREAL *table = toRArray(ttable)
    cdef cREAL *vec = allocR(vectorsize)

    for i in range(0, length, 2):
        v = lines[i]
        w = lines[i + 1]
        memset(vec, 0, vectorsize * 4)
        scopy(&vectorsize, &table[v * vectorsize], &iONE, vec, &iONE)
        saxpy(&vectorsize, &fmONE, &table[w * vectorsize], &iONE, vec, &iONE)
        vlength = snrm2(&vectorsize, vec, &iONE)
        if vlength > maxlength:
            maxlength = vlength
        break
    return maxlength


