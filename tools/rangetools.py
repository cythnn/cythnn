import os, math

from tools.taketime import taketime
from itertools import islice

# splits a range into #n subranges
def chunkRangeN(rnge, n):
    step = math.ceil(len(rnge) / n)
    return [ range(i, min(rnge.stop, i + step))
             for i in range(rnge.start, rnge.stop, step) ]

