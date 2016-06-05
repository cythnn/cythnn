# Describe a chunk to be processed by a specific pipe (pipeline module).
# pipeid: int that identifies the pipe to process
# priority: a task should have a priority between 0-1 (exclusive), where lower is a higher
#           priority. A task is prioritized first by a higher pipeid, then by a lower priority value.
# atiteration & atpipeid: preconditions that block processing a task until the progress arrived at (least) the given iteration and pipeid
# Tasks are full python objects, that hold additional attributes as parameters for the task.
class Task:
    def __init__(self, iteration=0, pipeid = 0, priority = 0, atiteration = None, atpipeid = None):
        self.pipeid = pipeid
        self.priority = priority
        self.iteration = iteration
        self.atiteration = atiteration
        self.atpipeid = atpipeid

    def isBlocked(self, learner):
        if (self.atiteration is None or self.atiteration <= learner.currentiteration):
            if (self.atpipeid is None or self.atpipeid == learner.currentpipeid or self.atiteration < learner.currentiteration):
                return False
        return True

    def __str__(self):
        return "Task(%s, %s)" % (self.pipeid, self.priority)

    def __lt__(self, other):
        return True if self.iteration + self.priority - self.pipeid < other.iteration + other.priority - other.pipeid else False

    def nextTask(self):
        return Task(iteration = self.iteration, priority = self.priority, pipeid=self.pipeid+1)
