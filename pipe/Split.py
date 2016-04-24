from pipe.pipe import Pipe
from model.learner import Task

# When split is used, tasks are formed that operate only on a disjoint set of words.
# Each thread has been assigned a taskid and only works tasks with their task id (and general tasks)
# to minimize memory collisions.
# In ths module, the task id duplicated for the given number of tasks, so that every
# task is operated on the entire input once.
class Split(Pipe):
    def __init__(self, pipeid, learner):
        Pipe.__init__(self, pipeid, learner)
        if hasattr(self.model, 'split'):
            self.split = self.model.split
        else:
            self.split = 0

    def isNeeded(self): # is removed when not in split mode
        return self.split == 1

    def feed(self, threadid, task):
        for taskid in range(self.getTaskids(task)):
            newtask = task.nextTask()
            newtask.words = task.words
            newtask.clower = task.clower
            newtask.cupper = task.cupper
            newtask.length = task.length
            newtask.taskid = taskid
            self.addTask(newtask)
