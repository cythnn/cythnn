import cython
from pipe.cy cimport cypipe

# just an example, a cypipe implementation should extend cypipe, for convenience followme is a typedef for the function arguments
# that will be used to pass data to its successor, and the class should have an attribute to store the method to call the next
# in the pipeline. Note that "nogil" must also match the methods declaration
ctypedef void(*outputMethod)(self, int value) nogil

# cypipe example must extend cypipe, and implement binTo() and bind().
# The parent class will register the threadid and model on construction
# The model will call bindTo() on the successor of a cypipe to register the successor
# and its process() method.
cdef class cypipe_example(cypipe):

    # to store the method of its successor, which must have the proper arguments.
    cdef outputMethod successorMethod

    # the method that does the actual processing, whose arguments must correspond to the output of its predecessor
    cdef void process(self, int value) nogil



