from model.solution cimport Solution
from libc.stdio cimport *

# cython variant of Pipe, which can be used to extend a Pipeline module in Cython
# A CPipe must override the feed() method to accept a task, and when overriding the
# constructor pass the parameters to the super constructor. In a CPipe, you can
# reference self.model, self.solution and self.learner.
cdef class CPipe:
    def __init__(self, pipeid, learner):
        if learner is None:
            print("Warning, cypipe created without a learner")
        else:
            self.learner = learner
            self.model = learner.model
            self.solution = self.model.getSolution()
        self.pipeid = pipeid
        self.build()

        setvbuf(stdout, NULL, _IONBF, 0);  # for debugging, turn off output buffering

    # is called before the init of the CPipe is executed, to allow to build required extensions to the model
    # typically, a learning architecture creates its own solution space in its build routine
    def build(self):
        pass

    def nextPipeId(self):
        return self.pipeid + 1

    # allows a Pipe to transform itself, based on the model's configuration it can
    # return None to remove itself from the Pipeline
    # return another Pipe instance to serve in its place
    def transform(self):
        return self

    # is the entry point for a Pipe to receive a Task that must be processed. In CPipes,
    # commonly the parameters are prepared in Python followed by a call to a Cython method
    # optionally followed by pushing new task(s) to the learner for processing by the
    # consecutive Pipe in the pipeline.
    def feed(self, task):
        raise NotImplementedError("A Pipe must implement the feed(task) method.")

    # adds a task to be processed by the consecutive (by default) Pipe in the pipeline
    def addTask(self, task):
        self.learner.addTask(task)

