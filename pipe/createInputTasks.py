from pipe.pipe import Pipe
from tools.wordio import inputDecay, inputUniform

class createW2VInputTasks(Pipe):
    def feed(self, threadid, task):
        if task.iteration < self.model.iterations - 1:  # for all but last iteration use uniform chunk sizes
            input = inputUniform(self.model, self.model.threads)
        else:  # for last iteration, use varying chunk sizes
            input = inputDecay(self.model, self.model.threads)
        for index, input in enumerate(input):
            newtask = task.nextTask()
            newtask.priority=1 / len(input.inputrange)
            newtask.inputstream = input
            self.addTask(newtask)

