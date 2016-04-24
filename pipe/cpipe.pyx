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

    # if a Pipe is optional and for the give configuration returns false, it is removed from the pipeline
    def isNeeded(self):
        return True

    # is the entry point for a Pipe to receive a Task that must be processed. In CPipes,
    # commonly the parameters are prepared in Python followed by a call to a Cython method
    # optionally followed by pushing new task(s) to the learner for processing by the
    # consecutive Pipe in the pipeline.
    def feed(self, taskid, task):
        raise NotImplementedError("A Pipe must implement the feed(taskid, task) method.")

    # the number of unique task id's (when no task ids are used the number of threads)
    def getTaskids(self, task):
        if self.model.split == 0 or task.iteration == 0:
            return self.model.tasks
        return self.model.tasks - self.solution.singletaskids


    # adds a task to be processed by the consecutive (by default) Pipe in the pipeline
    def addTask(self, task):
        self.learner.addTask(task)

