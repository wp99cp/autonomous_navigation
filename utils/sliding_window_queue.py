from queue import Queue

import numpy as np


class SlidingWindowQueue(Queue):
    def put(self, *args, **kwargs):
        if self.full():
            self.get()
        Queue.put(self, *args, **kwargs)

    def dump_as_array(self):

        arr = []
        while not self.empty():
            arr.append(self.get())

        assert len(arr) == self.maxsize and self.empty(), f"Expected {self.maxsize} items, got {len(arr)}"
        return np.array(arr)
