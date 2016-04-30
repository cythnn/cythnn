# return a term-frequncy dict of the given word-iterable
from multiprocessing.pool import Pool

from imdb.imdbstream import ImdbStream
from tools.worddict import Vocabulary, mergeDicts
from tools.wordio import inputUniform


def countWords(words):
    dict = {}
    imdbids = {}
    for word in words:
        try:
            imdbids[words.imdbid] += 1
        except KeyError:
            imdbids[words.imdbid] = 1
        try:
            dict[word] += 1
        except KeyError:
            dict[word] = 1
    return dict, imdbids

def buildvocab(learner, model):
    if not hasattr(model, 'vocab'):
        model.inputstreamclass = ImdbStream
        pool = Pool(processes=model.cores)
        tokens = pool.map(countWords, inputUniform(model, model.cores))
        words = mergeDicts([dict[0] for dict in tokens])
        imdbids = mergeDicts([dict[1] for dict in tokens])

        v = Vocabulary(words, model.mintf)
        model.setVocab(v)
        model.itemids = []
        model.indexeditems = {}
        for word, count in sorted( imdbids.items(), key=lambda x: -x[1]):
            model.indexeditems[word] = len(model.itemids)
            model.itemids.append(word)

        model.itemsize = len(model.itemids)
        #print([ (str(w), w.index) for w in model.vocab.sorted] )

        print("vocabulary build |v|=%d |i|=%d |c|=%ld"%(len(v), len(model.itemids), v.totalwords))
