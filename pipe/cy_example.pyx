from pipe.cy cimport cypipe
from model.cy cimport modelc
from libc.stdio cimport *

# example of a cypipe class, that is called by its process() method by its predecessor
# which calls the nextfunction(next, ...) to pass its results to its successor
cdef class cypipe_example(cypipe):

    # the constructor of cypipe can be overridden to prepare the module for processing, for instance casting
    # and setting static model parameters and resources. If no preparation is needed, a constructor is not
    # necessary
    def __init__(self, threadid, model):
        cypipe.__init__(self, threadid, model)

    # every cypipe needs to implement bindTo() and bind().
    # bindTo registers a cypipe instance to its predecessor by calling bind()
    # referencing itself and its process() method. If the process() arguments
    # do not match the predecessors output parameters, casting will fail during runtime.
    cdef void bindToCypipe(self, cypipe predecessor):
        predecessor.bind(self, <void*>(self.process))

    # bind registers references to the successor and its processing method. The processing
    # method should be cast to its correct type, and raise a clear TypeError when it does not
    # comply with the expected output
    cdef void bind(self, cypipe successor, void *method):
        self.successor = successor
        try:
            self.successorMethod = <outputMethod> method
        except:
            raise TypeError("Attempting to bind cypipe_example to a successor cypipe method that does not accept the correct argument types")

    # a cypipe class that can be used as the first class in the cypipe pipeline should have a Python
    # enterPipeline() method that allows calling it from Python, and converts the output from the last
    # pypipe module. The enterPipeline() method should always have a single argument feed, which can be
    # a tuple that has to be unpacked to match the c-arguments for its process methods.
    def feed(self, input):
        self.process(input)

    # every cypipe has a process() method, that should have arguments that match the output of its caller.
    # When the caller is expected to have release the GIL, the process method should also be in nogil mode.
    # The process method should call its successor when applicable (i.e. the last cypipe class usually knows
    # it is always last and therefor may not check whether there is a successor).
    cdef void process(self, int value) nogil:
        printf("cypipe_example %d %d\n", self.threadid, value)
        if self.successorMethod != NULL:
            self.successorMethod(self.successor, value + 1)

