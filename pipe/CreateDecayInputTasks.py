from pipe.pipe import Pipe
from tools.wordio import inputDecay, inputUniform

class createW2VDecayInputTasks(Pipe):
    def feed(self, threadid, task):
        input = inputDecay(self.model, self.model.threads * 2)
        for index, inp in enumerate(input):
            newtask = task.nextTask()
            newtask.priority=1 / len(inp.inputrange)
            newtask.inputstream = inp
            self.addTask(newtask)

