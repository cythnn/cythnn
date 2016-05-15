from contextlib import contextmanager
from queue import Queue
import threading

# helper class to lock a priority queue for the use of empty before get
class QueueSync(Queue):
    def __init__(self):
        Queue.__init__(self)
        self._lock = threading.Lock()

    # need to lock to use empty on shared priorityqueue
    def get(self):
        with self.acquire_timeout():
            if not self.empty():
                return Queue.get(self)
            else:
                return None

    @contextmanager
    def acquire_timeout(self):
        result = self._lock.acquire()
        yield result
        if result:
            self._lock.release()
