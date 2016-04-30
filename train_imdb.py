from datetime import datetime

from imdb.ParagraphVector import ParagraphVector
from imdb.imdbvocab import buildvocab
from imdb.ConvertWordIds import convertWordIds
from imdb.createInputTasks import createImdbInputTasks
from imdb.save import save
from model.model import Model
from tools.taketime import taketime


def ImdbModel(inputrange=None):
    return Model(alpha=0.025, vectorsize=100,
                 input="data/imdb.0",
                 inputrange=inputrange,
                 build=[ buildvocab ],
                 pipeline=[ createImdbInputTasks, convertWordIds, ParagraphVector ],
                 mintf=5, cores=2, threads=3, iterations=10,
                 downsample=0.001, updaterate=100, innercache=0
                 )

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = ImdbModel()
    #m = ImdbModel(inputrange=range(1000000))
    time(m)
    save("results/imdb.bin", m)
    print("done", str(datetime.now()))

