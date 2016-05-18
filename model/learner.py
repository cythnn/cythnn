from __future__ import print_function
from contextlib import contextmanager
from queue import PriorityQueue
from time import time, sleep
import threading

from datetime import datetime
import numpy as np

# Orchestrates the learning process, by starting the required number of threads, managing the
# PriorityQueue that contains the tasks to be processed, and reporting progress.
from model.task import Task
from tools.priorityQueueSync import PriorityQueueSync

class Learner:
    def __init__(self, model):
        self.model = model
        self.threads = model.threads
        self.iterations = model.iterations
        self.tasks = model.tasks
        self.pipe = [None for x in range(len(model.pipeline))] # first pipelineobject per thread

    # Builds the vocabulay, then build the learning pipeline, and push the inputs through the pipeline.
    def run(self):
        jobtime = time()
        # queues for the job
        self.queue = [ PriorityQueueSync() for i in range(self.tasks) ]
        self.generalqueue = PriorityQueueSync()
        self.finished = set()

        # build the model
        for f in self.model.build:
            f(self, self.model)

        if self.model.quiet == 0:
            print("preprocessing finished %0.2f sec"%(time() - jobtime))

        # instantiate the processing pipeline
        self.createPipes()

        # creates the shared solution space
        solution = self.model.getSolution()

        # creates tasks as input for the first module in the pipeline, given the input and number of threads
        self.setupTasksIterations()

        if self.model.quiet == 0:
            print("running multithreadeded threads %d iterations %d" %
            ( self.threads, self.iterations))

        threads = []
        for threadid in range(self.threads):
            taskid = threadid   # extensions can assign multiple threads to the same task id
            t = threading.Thread(target=learnThread, args=(threadid, taskid, self))
            t.daemon = True
            threads.append(t)
            t.start()

        starttime = time()
        while len(self.finished) < self.threads:
            sleep(2)   # update every 2 seconds
            p = solution.getProgressPy();
            if p > 0 and self.model.quiet == 0:
                wps = self.getTotalWords() * self.iterations * p / (time() - starttime)
                alpha = solution.getCurrentAlpha()
                if self.model.quiet == 0:
                    print("progress %4.1f%% wps %d alpha %f\n" % (100 * p, int(wps), alpha), end = '')
                #print(self.activeThreads())
        if self.model.quiet == 0:
            print("\ndone %0.1f sec"%(time() - jobtime))

    # fro debugging purposes, see which threads are active
    def activeThreads(self):
        s = ""
        for i in range(self.threads):
            s += "0" if i in self.finished else "1"
        return s

    def getTotalWords(self):
        return self.model.vocab.totalwords

    # add a task for every iteration, commonly the first task in the pipeline sets up the input
    def setupTasksIterations(self):
        for iter in range(self.iterations):
            self.addTask(Task(iteration=iter))

    def addTask(self, task):
        if task.taskid is None:
            self.generalqueue.put(task)
        else:
            self.queue[task.taskid].put(task)

    def getTask(self, threadid, taskid):
        task = self.queue[taskid].get()
        if task is None:
            task = self.generalqueue.get()
        return task

    def createPipes(self):
        pipeid = 0
        for i in range(len(self.model.pipeline)):
            p = self.model.pipeline[i](pipeid, self)

            # a Pipe may remove (when it is not needed) or replace (e.g. a factory) itself
            n = None
            while p is not None and n != p:
                n = p
                p = p.transform()

            # add to pipeline when not removed
            if p is not None:
                self.pipe[pipeid] = p
                pipeid += 1

        # print the final pipeline that is used
        if self.model.quiet == 0:
            print("pipeline", [self.pipe[i].__class__.__name__ for i in range(pipeid)])

# a thread has a designated taskid, and repeatedly picks task (matching its desgnated taskid
# or a general task), and processes it using the  queue and calls
# the trainer on that chunk, until there is no more input
def learnThread(threadid, taskid, learner):
    while len(learner.finished) < learner.threads: # pick a task and process
        task = learner.getTask(threadid, taskid)
        if task is not None:
            if threadid in learner.finished:
                learner.finished.remove(threadid)
            learner.pipe[task.pipeid].feed(threadid, task)
        else:
            learner.finished.add(threadid)
            sleep(0.1)
