from model.solution cimport Solution

# cython variant of Pipe, which can be used to extend a Pipeline module in Cython
# A CPipe must override the feed() method to accept a task, and when overriding the
# constructor pass the parameters to the super constructor. In a CPipe, you can
# reference self.model, self.solution and self.learner.
cdef class CPipe:
    def __init__(self, taskid, learner):
        if learner is None:
            print("Warning, cypipe created without a learner")
        else:
            self.learner = learner
            self.model = learner.model
            self.solution = self.model.getSolution()
        self.taskid = taskid

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
    def addTask(self, task, oldtask, pipeid = None):
        task.iteration = oldtask.iteration
        if pipeid is None:
            task.pipeid = oldtask.pipeid + 1
        else:
            task.pipeid = pipeid
        self.learner.addTask(task)

