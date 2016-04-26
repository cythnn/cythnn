from __future__ import print_function
from model.learner import Learner
from model.solution import Solution
from tools.wordio import wordStreamsDecay, wordStreams

# Contains the configuration for the network, and any module that is used in it
class Model:
    def __init__(self,  input,              # flat file with text used for learning
                        inputrange=None,    # uses only the given range in teh input file
                        alpha=0.025,        # initial learning rate
                        build=[],           # functions called with (model) to construct the model
                        pipeline=[],        # python Pipe classes that process the input word generator
                        mintf=1,            # minimal collection frequency for terms to be used
                        windowsize=5,       # windowsize used for w2v models
                        iterations=1,       # number of times to iterate over the corpus when learning
                        threads=None,       # hard defines the number of threads used for learning, otherwise cores is used
                        cores=2,            # defines number of threads used for building
                        updaterate=100,     # after this number of processed words, cached vectors, processed words and alpha are updated
                        wordcache=0,        # number of most frequent words to cache to avoid memory collisions between threads
                        innercache=100,     # number of most frequent inner nodes to cache to avoid memory collisions between threads
                        downsample=0,        # parameter for downsampling frequent terms (0=no downsampling)
                        **kwargs):
        self.__dict__.update(kwargs)
        self.input = input;
        self.inputrange = inputrange
        self.alpha = alpha
        self.build = build
        self.pipeline = pipeline
        self.mintf = mintf
        self.windowsize = windowsize
        self.iterations = iterations
        self.updaterate = updaterate        # set to 10k when not caching
        self.wordcache = wordcache
        self.innercache = innercache
        self.downsample = downsample        # typical settings: 0, 10e-3 or 10e-5

        # number of cores/threads to use in multithreading mode, by default for every core two
        # threads are used to overcome performance loss by memory blocks
        self.threads = threads if threads is not None else cores
        self.cores = cores if cores is not None else self.threads
        self.tasks = self.threads

    # Builds the vocabulay, then build the learning pipeline, and push the inputs through the pipeline.
    def run(self):
        Learner(self).run()

    # Used by vocabulary builders to store the result in the Model
    def setVocab(self, vocab):
        self.vocab = vocab
        # number of unique words in vocab, vocab.totalwords contains the total count
        self.vocsize = len(vocab)
        # output size of the w2v model, can be modified by other modules (e.g. HS trains against a Huffmann tree instead of the vocabulary)
        self.outputsize = len(vocab)

    def setSolution(self, matrices):
        self.matrices = matrices                    # must store a Python reference to prevent garbage collection!
        self.getSolution().setSolution(matrices)    # since the solution is in Cython

    # return the solution, which is instantiated on first request, which should be after the vocabulary was build.
    def getSolution(self):
        if not hasattr(self, 'solution'):
            self.solution = Solution(self)
        return self.solution
