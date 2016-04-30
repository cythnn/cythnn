import numpy as np
from numpy import float32

def readmovieids(fname):
    with open(fname, 'r') as fin:
        fin.readline()
        movies = {}
        for line in fin.readlines():
            terms = line.split(' ')
            movies[terms[0]] = terms[]
        return movies

def tableimdbml(movies):
    return { movie[2]:movie[0] for movie in movies.items() }

def readratings(fname):
    with open(fname, 'r') as fin:
        fin.readline()
        users = {}
        for line in fin.readlines():
            terms = line.split(' ')
            try:
                user = users.get(terms[0])
            except KeyError:
                user = []
                users[terms[0]] = user
            user.append((terms[1], terms[2]))
        return users

def readembeddings(fname, movies):
    table = tableimdbml(movies)
    items = dict()
    with open(fname, 'r') as fin:
        itemcount, vectorsize = fin.readline().split(" ")
        items = dict(itemcount)
        embeddings = np.zeros((itemcount, vectorsize), dtype=float32)
        index = 0
        for line in fin.readlines():
            terms = line.split(' ')
            imdbid = terms[0]
            mlid = table[imdbid]
            for i in range(vectorsize):
                embeddings[index, i] = terms[i + 1]
            items[mlid] = embeddings[index]
            index += 1
        return embeddings

if __name__ == "__main__":
    userratings=readratings()
    for user, ratings in userratings.items():



