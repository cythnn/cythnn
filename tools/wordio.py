import os, math, re

# wraps a generator of words read from a flat text #file, within position #byterange
# uses virtual soft partitioning of flat text files, a partition starts after the first whitespace
# and the prior partition reads until the first word seperator after the boundary
from multiprocessing.pool import Pool

from tools.taketime import taketime

# return filesize
def size(path):
    return os.path.getsize(path)

def chunkRangeS(rnge, size):
    return [ range(i, min(rnge.stop, i + size))
             for i in range(rnge.start, rnge.stop, size) ]

class WordStream:
    def __init__(self, byterange=None, file=None, window = 0):
        self.file = file
        if file and byterange is None:
            self.range = range(0, size(file))
        else:
            self.range = byterange
        self.window = window
        self.wentBack = 0
        self.wentPast = -1

    def readFirst(self, f, bytepos, end):
        self.wentBack = 0
        start = max(0, min(bytepos, bytepos - 100 * self.window))
        end = max(bytepos, bytepos + 1000000)
        if start > 0:
            f.seek(start)
        buffer = f.read(end - start)
        pos = bytepos - start
        if self.window > self.wentBack and bytepos > 0:
            while pos > 0 and self.window > self.wentBack:
                pos -= 1
                if buffer[pos] == ' ' or buffer[pos] == '\n':
                    self.wentBack += 1
                    if self.window == self.wentBack:
                        buffer = buffer[pos+1:]
            if pos == 0:
                self.wentBack += 1
        elif bytepos > 0:
            while pos < len(buffer):
                if buffer[pos] == ' ' or buffer[pos] == '\n':
                    break
                pos += 1
            buffer = buffer[pos+1:]
        return buffer

    def __iter__(self):
        buffer = ""
        with open(self.file, "r") as f:
            for chunk in chunkRangeS(self.range, 1000000):
                if chunk.start == self.range.start and chunk.start > 0:
                    buffer = self.readFirst(f, chunk.start, chunk.stop)
                else:
                    newbuf = f.read(chunk.stop - chunk.start)
                    if not newbuf:
                        yield buffer
                        yield "</s>"
                        buffer = ""
                        break
                    buffer += newbuf
                for sentence in re.split('(\n)', buffer):
                    if sentence == '\n':
                        yield buffer
                        yield '</s>'
                        buffer = ""
                    else:
                        words = sentence.split(' ')
                        for word in words[:-1]:
                            yield word
                        buffer = words[-1]
            newbuf = f.read((self.window + 1) * 100)
            if not newbuf:
                self.wentPast += 1
                if len(buffer) > 0:
                    yield buffer
            else:
                buffer += newbuf
                for sentence in re.split('(\n)', buffer):
                    for word in sentence.split(' '):
                        self.wentPast += 1
                        yield word
                        if self.window <= self.wentPast:
                            return
                    self.wentPast += 1
                    yield '</s>'
                    return

#setup a list of #parts WordStream objects, that cover the given #byterange
@taketime("wordstreams")
def wordStreams(path, parts = 2, byterange = None, window = 0):
    if byterange is None:
        byterange = range(0, size(path))
    return [WordStream(r, path, window=window)
            for r in chunkRange(byterange, parts)]

#split range in #n consecutive sub-ranges
def chunkRange(rnge, n):
    step = math.ceil(len(rnge) / n)
    return [ range(i, min(rnge.stop, i + step))
             for i in range(rnge.start, rnge.stop, step) ]

