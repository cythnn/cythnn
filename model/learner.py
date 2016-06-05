from __future__ import print_function
from contextlib import contextmanager
from queue import PriorityQueue
from time import time, sleep
import threading

from datetime import datetime
import numpy as np

# Orchestrates the learning process, by starting the required number of threads, managing the
# tasks to be processed, and reporting progress.

class Learner:
    def __init__(self, model):
        self.model = model
        self.threads = model.threads
        self.iterations = model.iterations
        self.tasks = model.tasks
        self._lock_tasks = threading.Lock()

    # Builds the vocabulay, then build the learning pipeline, and push the inputs through the pipeline.
    def run(self):
        self.jobstarttime = time()
        # queues for the job
        self.__queue = PriorityQueue()
        self.__blockedtasks = []
        self.currentiteration = 0
        self.currentpipeid = 0

        # build the model
        for f in self.model.build:
            f(self, self.model)

        if self.model.quiet == 0:
            print("preprocessing finished %0.2f sec"%(time() - self.jobstarttime))

        # instantiate the processing pipeline
        self.__progress = np.zeros((self.iterations, len(self.model.pipeline)), dtype=int)
        self.createPipes()

        solution = self.model.getSolution()

        if self.model.quiet == 0:
            print("running multithreadeded threads %d iterations %d" %
            ( self.threads, self.iterations))

        threads = []
        self.inactivethreads = set()
        for threadid in range(self.threads):
            t = threading.Thread(target=learnThread, args=(threadid, self))
            t.daemon = True
            threads.append(t)
            t.start()

        starttime = time()
        while not self.finished():
            sleep(2)   # update every 2 seconds
            p = solution.getProgressPy();
            if p > 0 and self.model.quiet == 0:
                wps = self.getTotalWords() * self.iterations * p / (time() - starttime)
                alpha = solution.getCurrentAlpha()
                print("progress %4.1f%% wps %d alpha %f\n" % (100 * p, int(wps), alpha), end = '')
                #print(self.activeThreads())
        if self.model.quiet == 0:
            print("\ndone %0.2f sec"%(time() - self.jobstarttime))

    # fro debugging purposes, see which threads are active
    def activeThreads(self):
        s = ""
        for i in range(self.threads):
            s += "0" if i in self.inactivethreads else "1"
        return s

    def getTotalWords(self):
        return self.model.vocab.totalwords

    def addTask(self, task):
        with self._lock_tasks:
            if task.isBlocked(self):
                self.__blockedtasks.append(task)
            else:
                self.__queue.put(task)
            self.__progress[task.iteration][task.pipeid] += 1

    def getTask(self, threadid):
        with self._lock_tasks:
            if self.__queue.qsize() > 0:
                return self.__queue.get()
            return None

    def finishedTask(self, threadid, task):
        with self._lock_tasks:
            #print("finished iter %d pipe %d"%( self.currentiteration, self.currentpipeid))
            self.__progress[task.iteration][task.pipeid] -= 1
            if task.iteration == self.currentiteration and self.currentpipeid == task.pipeid and self.__progress[task.iteration][task.pipeid] == 0:
                while self.currentiteration < self.iterations and self.__progress[self.currentiteration][self.currentpipeid] == 0:
                    if self.currentpipeid < len(self.pipe) - 1:
                        self.currentpipeid += 1
                    else:
                        self.currentiteration += 1
                        self.currentpipeid = 0
                #print("push iter %d pipe %d"%( self.currentiteration, self.currentpipeid))
                for t in self.__blockedtasks:
                    if not t.isBlocked(self):
                        self.__queue.put(t)
                self.__blockedtasks[:] = [ t for t in self.__blockedtasks if t.isBlocked(self) ]

    def finished(self):
        return self.currentiteration >= self.iterations

    def createPipes(self):
        self.pipe = []
        pipeid = 0
        for i in range(len(self.model.pipeline)):
            p = self.model.pipeline[i](pipeid, self)
            n = None
            while p is not None and n != p:
                n = p
                p = p.transform() # a Pipe may remove or replace itself

            if p is not None:
                self.pipe.append(p)
                pipeid += 1
        if self.model.quiet == 0:
            print("pipeline", [p.__class__.__name__ for p in self.pipe])

# a thread has a designated taskid, and repeatedly picks task (matching its desgnated taskid
# or a general task), and processes it using the  queue and calls
# the trainer on that chunk, until there is no more input
def learnThread(threadid, learner):
    while not learner.finished(): # pick a task and process
        task = learner.getTask(threadid)
        if task is not None:
            if threadid in learner.inactivethreads:
                learner.inactivethreads.remove(threadid)
            learner.pipe[task.pipeid].feed(threadid, task)
            learner.finishedTask(threadid, task)
        else:
            learner.inactivethreads.add(threadid)
            sleep(0.1)
