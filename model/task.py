# Describe a chunk to be processed by a specific pipe (pipeline module).
# taskid: when given, assigns the task only to threads that have the same taskid,
#         otherwise, any thread my take the task
# pipeid: int that identifies the pipe to process
# priority: a task should have a priority between 0-1 (exclusive), where lower is a higher
#           priority. A task is prioritized first by a higher pipeid, then by a lower priority value.
# Tasks are full python objects, that hold additional attributes as parameters for the task.
class Task:
    def __init__(self, iteration=0, taskid = None, pipeid = 0, priority = 0):
        self.taskid = taskid
        self.pipeid = pipeid
        self.priority = priority
        self.iteration = iteration

    def __str__(self):
        return "Task(%s, %s, %s)" % (self.taskid, self.pipeid, self.priority)

    def __lt__(self, other):
        return True if self.iteration + self.priority - self.pipeid < other.iteration + other.priority - other.pipeid else False

    def nextTask(self):
        return Task(iteration = self.iteration, priority = self.priority, pipeid=self.pipeid+1, taskid=self.taskid)
