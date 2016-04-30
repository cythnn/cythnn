from imdb.imdbstream import ImdbStream
from pipe.pipe import Pipe
from tools.wordio import inputUniform, inputDecay

class createImdbInputTasks(Pipe):
    def feed(self, threadid, task):
        self.model.inputstreamclass = ImdbStream
        if task.iteration < self.model.iterations - 1:  # for all but last iteration use uniform chunk sizes
            input = inputUniform(self.model, self.model.threads * 2)
        else:  # for last iteration, use varying chunk sizes
            input = inputDecay(self.model, self.model.threads * 2)
        for index, input in enumerate(input):
            newtask = task.nextTask()
            newtask.priority=1 / len(input.inputrange)
            newtask.inputstream = input
            self.addTask(newtask)
