from contextlib import contextmanager
from queue import PriorityQueue
import threading

# helper class to lock a priority queue for the use of empty before get
class PriorityQueueSync(PriorityQueue):
    def __init__(self):
        PriorityQueue.__init__(self)
        self._lock = threading.Lock()

    # need to lock to use empty on shared priorityqueue
    def get(self):
        with self._lock:
            if not self.empty():
                return PriorityQueue.get(self)
            else:
                return None

