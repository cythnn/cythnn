from model.cy cimport modelc

# cypipes consist of objects, that perform a processing task and pass on their results to the next
# object in the pipeline. A benefit of using objects, is that it allows these objects to remember their state,
# however, they are slightly more work to setup than functions in cython. A cypipe class must extend CYPIPE
# implement a bind(self, cypipe next, void *nextfunction) and a bindTo(self, cypipe predecessor).
cdef class cypipe:
    cdef modelc modelc
    cdef int threadid
    cdef cypipe successor

    # only for use in bindTo, binds a cypipe to its successor in the pipeline
    cdef void bind(self, cypipe successor, void *successorMethod)

    # used when consructing a pipeline, should call bind on its predecessor with itself and the function it
    # uses to process the pipeline data. The function arguments shoudl match the arguments that are passed
    # by the predecessor class
    cdef void bindToCypipe(self, cypipe predecessor)





