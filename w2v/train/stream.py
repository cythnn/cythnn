import cython
from numpy import uint64, ndarray, errstate

from tools.taketime import taketime

rand = uint64(25214903917)
c11 = uint64(11)

def stream(vocab, str, windowsize):
    next_random = uint64(1);
    words = list(genWords(vocab, str))
    for i, word in enumerate(words):
        if i < str.wentBack or i >= len(words) - str.wentPast: continue
        with errstate(over='ignore'):
            next_random = next_random * rand + c11;
        b = int(next_random) % windowsize
        for c in range(max(i - windowsize + b, 0), min(len(words), i + windowsize + 1 - b)):
            if c != i:
                last_word = words[c]
                if last_word.index == 0: break
                for (exp, inner) in word.innernodes:
                    yield last_word.index, inner.index, exp

@taketime("genwords")
def genWords(vocab, str):
    for term in str:
        word = vocab.get(term)
        if word is not None:
            yield word
