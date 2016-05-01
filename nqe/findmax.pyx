from numpy import zeros, float32, int32, dtype, fromstring
from tools.types cimport *
from numpy cimport *
from tools.blas cimport saxpy, scopy, snrm2

def loadvectors(fname, length=None):
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
        for line_no in range(length if length is not None else vocab_size):
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

cdef cREAL findmax(ndarray llines, ndarray ttable):
    cdef int v, w, i, length = len(llines)
    cdef int vectorsize = ttable.shape[1]
    cdef cREAL vlength
    cdef cREAL maxlength = 0
    cdef cINT *lines = toIArray(llines)
    cdef cREAL *table = toRArray(ttable)
    cdef cREAL *vec = allocR(vectorsize)

    for i in range(0, length, 2):
        v = lines[i]
        w = lines[i + 1]
        scopy(&vectorsize, &table[v * vectorsize], &iONE, vec, &iONE)
        saxpy(&vectorsize, &fmONE, &table[w * vectorsize], &iONE, vec, &iONE)
        vlength = snrm2(&vectorsize, vec, &iONE)
        if vlength > maxlength:
            maxlength = vlength
        break
    free(vec)
    return maxlength


cdef void write(object file, ndarray resultv, ndarray resultn, ndarray results, int length):
    for i in range(length):
        pair=' '.join("%d" % val for val in results[i])
        vector =' '.join("%f" % val for val in resultv[i])
        file.write("%s %f %s\n" % (pair, resultn[i], vector))

cdef void cluster(ndarray ttable, ndarray vtable, ndarray ntable, ndarray itable, int partition, int partitions, float maxlength):
    cdef int m0, m1, v, w, i, rc = 0
    cdef float mv0, mv1, a0, a1
    cdef int length = ttable.shape[0]
    cdef int vectorsize = ttable.shape[1]
    cdef cREAL *table = toRArray(ttable)
    cdef cREAL *vec = allocR(vectorsize)
    cdef cINT *results = toIArray(itable)
    cdef cREAL *resultv = toRArray(vtable)
    cdef cREAL *resultn = toRArray(ntable)

    cdef object file = open("out.%d"%partition, "w")

    for v in range(1, length):
        for w in range(v):
            if (v + w % partitions == partition):
                scopy( & vectorsize, & table[v * vectorsize], & iONE, vec, & iONE)
                saxpy( & vectorsize, & fmONE, & table[w * vectorsize], & iONE, vec, & iONE)
                vlength = snrm2( & vectorsize, vec, & iONE)
                if vlength < maxlength:
                    a0 = abs(vec[0])
                    a1 = abs(vec[1])
                    if a0 > a1:
                        mv0 = a0
                        m0 = 0
                        mv1 = a1
                        m1 = 1
                    else:
                        mv0 = a1
                        m0 = 1
                        mv1 = a0
                        m1 = 0
                    for i in range(2, vectorsize):
                        a0 = abs(vec[i])
                        if a0 > mv1:
                            if a0 > mv0:
                                mv1 = mv0
                                m1 = m0
                                mv0 = a0
                                m0 = i
                            else:
                                mv1 = a0
                                m1 = i
                    results[rc * 4] = v
                    results[rc * 4 + 1] = w
                    results[rc * 4 + 2] = m0
                    results[rc * 4 + 3] = m1
                    resultn[rc] = vlength
                    scopy( &vectorsize, vec, &iONE, &resultv[rc * vectorsize], &iONE)
                    rc = rc + 1
                    if (rc >= 10000):
                        if True:#with gil:
                           write(file, vtable, ntable, itable, rc)
                        rc = 0
    write(file, vtable, ntable, itable, rc)
    file.close()





