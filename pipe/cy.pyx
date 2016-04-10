from model.cy cimport modelc
from pipe.py import pypipe

# cypipes consist of objects, that perform a processing task and pass on their results to the next
# object in the pipeline. A benefit of using objects, is that it allows these objects to remember their state,
# however, they are slightly more work to setup than functions in cython. A cypipe class must extend CYPIPE
# implement a bind(self, cypipe next, void *nextfunction) and a bindTo(self, cypipe predecessor).
cdef class cypipe:
    def __init__(self, threadid, model):
        if model is None:
            print("Warning, cypipe created without a model")
        else:
            self.modelc = model.getModelC()
        self.threadid = threadid

    # only for use in bindTo, binds a cypipe to its successor in the pipeline
    cdef void bind(self, cypipe successor, void *successorMethod):
        raise NotImplementedError("Must override bind to register the processing function of a cypipe predecessor")

    # used when consructing a pipeline, should call bind on its predecessor with itself and the function it
    # uses to process the pipeline data. The function arguments shoudl match the arguments that are passed
    # by the predecessor class
    def bindTo(self, predecessor):
        if isinstance(predecessor, cypipe):
            self.bindToCypipe(predecessor)
        elif isinstance(predecessor, pypipe):
            predecessor.bind(self)
        else:
            raise TypeError("Can only use modules in the pipeline that extend pypipe or cypipe")

    cdef void bindToCypipe(self, cypipe predecessor):
        raise NotImplementedError("Must override cdef void bindToCypipe(self, cypipe) to register a cypipe predecessor")

