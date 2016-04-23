from __future__ import print_function
from contextlib import contextmanager
from queue import PriorityQueue
import time
import threading
import numpy as np

# Orchestrates the learning process, by starting the required number of threads, managing the
# PriorityQueue that contains the tasks to be processed, and reporting progress.
from tools.wordio import wordStreams, wordStreamsDecay


class Learner:
    def __init__(self, model):
        self.model = model
        self.threads = model.threads
        self.iterations = model.iterations
        self.tasks = model.tasks
        self.progress = np.zeros((self.threads + 1), dtype=np.int32)  # tracks progress per thread
        self.pipe = [None for x in range(len(model.pipeline))] # first pipelineobject per thread

    # Builds the vocabulay, then build the learning pipeline, and push the inputs through the pipeline.
    def run(self):

        print("preprocessing start")
        # build the model
        for f in self.model.build:
            f(self, self.model)

        print("preprocessing finished")

        # instantiate the processing pipeline
        self.createPipes()

        solution = self.model.getSolution()

        # queues for the job
        self.queue = [ PriorityQueueSync() for i in range(self.tasks) ]
        self.generalqueue = PriorityQueueSync()
        self.finished = set()

        self.setupInputTasks()

        print("running multithreadeded threads %d parts %d iterations %d" %
        ( self.threads, len(self.model.input), self.iterations))

        threads = []
        for threadid in range(self.threads):
            taskid = threadid   # extensions can assign multiple threads to the same task id
            t = threading.Thread(target=learnThread, args=(threadid, taskid, self))
            t.daemon = True
            threads.append(t)
            t.start()

        starttime = time.time()
        while len(self.finished) < self.threads:
            time.sleep(2)   # update every 2 seconds
            p = solution.getProgressPy();
            if p > 0:
                wps = self.getTotalWords() * self.iterations * p / (time.time() - starttime)
                alpha = solution.getCurrentAlpha()
                print("progress %4.1f%% wps %d alpha %f\n" % (100 * p, int(wps), alpha), end = '')
                print(self.activeThreads())
            else:
                starttime = time.time()
                wps = 0
        print("\ndone")

    # fro debugging purposes, see which threads are active
    def activeThreads(self):
        s = ""
        for i in range(self.threads):
            s += "0" if i in self.finished else "1"
        return s

    def getTotalWords(self):
        return self.model.vocab.totalwords

    def addTask(self, task):
        #print("addtask", task.priority, task.pipeid, task)
        if task.taskid is None:
            self.generalqueue.put(task)
        else:
            self.queue[task.taskid].put(task)

    def getTask(self, threadid, taskid):
        task = self.queue[taskid].get()
        if task is None:
            task = self.generalqueue.get()
        return task

    # add all inputs as tasks, priorize largest first
    def setupInputTasks(self):
        for iteration in range(self.iterations):
            if iteration < self.iterations - 1:  # for all but last iteration use uniform chunk sizes
                input = self.getUniformInputs()
            else:  # for last iteration, use varying chunk sizes
                input = wordStreamsDecay(self.model.input, parts=self.threads, inputrange=self.model.inputrange,
                                         windowsize=self.model.windowsize)
            for index, input in enumerate(input):
                task = Task(iteration=iteration, priority=1 / len(input.inputrange))
                task.pyparams = input
                self.addTask(task)

    # add all inputs as tasks, priorize largest first
    def getUniformInputs(self):
        return wordStreams(self.model.input, parts=self.threads, inputrange=self.model.inputrange, windowsize=self.model.windowsize)

    def createPipes(self):
        for pipeid in range(len(self.model.pipeline)):
            self.pipe[pipeid] = self.model.pipeline[pipeid](pipeid, self)

# a thread has a designated taskid, and repeatedly picks task (matching its desgnated taskid
# or a general task), and processes it using the  queue and calls
# the trainer on that chunk, until there is no more input
# note must be a 
def learnThread(threadid, taskid, learner):
    while len(learner.finished) < learner.threads: # pick a task and process
        task = learner.getTask(threadid, taskid)
        if task is not None:
            if threadid in learner.finished:
                learner.finished.remove(threadid)
            learner.pipe[task.pipeid].feed(threadid, task)
        else:
            learner.finished.add(threadid)
            time.sleep(0.1)

# helper class to lock a priority queue for the use of empty before get
class PriorityQueueSync(PriorityQueue):
    def __init__(self):
        PriorityQueue.__init__(self)
        self._lock = threading.Lock()

    # need to lock to use empty on shared priorityqueue
    def get(self):
        with self.acquire_timeout():
            if not self.empty():
                return PriorityQueue.get(self)
            else:
                return None

    @contextmanager
    def acquire_timeout(self):
        result = self._lock.acquire()
        yield result
        if result:
            self._lock.release()

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