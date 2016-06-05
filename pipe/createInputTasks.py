from model.task import Task
from pipe.pipe import Pipe
from tools.wordio import inputDecay, inputUniform
from functools import partial

class createW2VInputTasks(Pipe):
    def feed(self, threadid, task):
        if task.iteration < self.model.iterations - 1 and not self.model.blockedmode:  # for all but last iteration use uniform chunk sizes
            input = [self.model.inputstreamclass(c[0], self.model.windowsize, c[1]) for c in inputUniform(self.model.input, self.model.threads)]
        else:  # for last iteration, use varying chunk sizes
            input = [self.model.inputstreamclass(c[0], self.model.windowsize, c[1]) for c in inputDecay(self.model.input, self.model.threads)]
        for index, inp in enumerate(input):
            newtask = task.nextTask()
            newtask.priority=1 / len(inp.inputrange)
            newtask.inputstream = inp
            self.addTask(newtask)

    # allows to start reading I/O for only the next iteration
    def build(self):
        for iter in range(self.model.iterations):
            self.addTask(Task(iteration=iter, pipeid=self.pipeid, atiteration=iter-1))


