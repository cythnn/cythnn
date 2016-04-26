from pipe.pipe import Pipe
from model.learner import Task

# Duplicates a task so that multiple tasks/threads operate on the same data
class Split(Pipe):
    def __init__(self, pipeid, learner):
        Pipe.__init__(self, pipeid, learner)

    def feed(self, threadid, task):
        for taskid in range(self.getTaskids(task)):
            newtask = task.nextTask()
            newtask.words = task.words
            newtask.clower = task.clower
            newtask.cupper = task.cupper
            newtask.length = task.length
            newtask.taskid = taskid
            self.addTask(newtask)
