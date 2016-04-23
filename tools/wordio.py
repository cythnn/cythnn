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

# reads the words that are in a flat text file, by a space or newline.
# The byterange indicates thedesignated range of words to be read.
# When the upper byterange boundary is set within a word or its first trailing word-separator
# the word is considered to be part of the range, when the lower byterange boundary is set
# within a word or its first trailing word-separator it is not.
# when window is set, an attempt is made to read #window words before and after
# the designated byterange, wentBack and wentPast indicate the number of words read before and
# after the designated boundary (max #window)
class WordStream:
    def __init__(self, inputrange=None, file=None, windowsize = 0):
        self.file = file
        if file and inputrange is None:
            self.inputrange = range(0, size(file))
        else:
            self.inputrange = inputrange
        self.windowsize = windowsize

    def readFirst(self, f, bytepos, end):
        start = max(0, min(bytepos, bytepos - 100 * self.windowsize))
        end = min(end, bytepos + 1000000)
        if start > 0:
            f.seek(start)
        buffer = f.read(end - start)
        pos = bytepos - start
        if self.windowsize > self.wentBack and bytepos > 0:
            while pos > 0 and self.windowsize > self.wentBack:
                pos -= 1
                if buffer[pos] == ' ' or buffer[pos] == '\n':
                    self.wentBack += 1
                    if self.windowsize == self.wentBack:
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
        self.wentPast = -1
        self.wentBack = 0
        with open(self.file, "r") as f:
            for chunk in chunkRangeS(self.inputrange, 1000000):
                if chunk.start == self.inputrange.start and chunk.start > 0:
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
            newbuf = f.read((self.windowsize + 1) * 100)
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
                        if self.windowsize <= self.wentPast:
                            return
                    self.wentPast += 1
                    yield '</s>'
                    return

#setup a list of #parts WordStream objects, that cover the given #byterange
#@taketime("wordstreams")
def wordStreams(path, parts = 2, inputrange = None, windowsize = 0, iterations = 1):
    if iterations < 1:
        return []
    if inputrange is None:
        inputrange = range(0, size(path))
    a = [WordStream(r, path, windowsize=windowsize)
            for r in chunkRange(inputrange, parts)]
    a.extend(wordStreams(path, parts, inputrange, windowsize, iterations - 1))
    return a

#@taketime("wordstreams")
def wordStreamsDecay(path, parts = 2, inputrange = None, windowsize = 0, iterations = 1):
    if inputrange is None:
        inputrange = range(0, size(path))
    a = [WordStream(r, path, windowsize=windowsize)
         for r in chunkRangeDecay(inputrange, parts)]
    a.extend(wordStreams(path, parts, inputrange, windowsize, iterations - 1))
    return a

#split range in #n consecutive sub-ranges
def chunkRange(rnge, n):
    step = math.ceil(len(rnge) / n)
    a = [ range(rnge.start + i * step, rnge.start + (i + 1) * step)
         for i in range(n - 1) ]
    a.append(range(rnge.start + (n - 1) * step, rnge.stop))
    return a

#split range in #n consecutive sub-ranges
def chunkRangeDecay(rnge, n):
    step = math.ceil( 2 * (rnge.stop - rnge.start) / n / (n + 1))
    a = []
    start = rnge.start
    for incr in range(step, (n) * step, step):
        a.append(range(start, start + int(incr)))
        start += int(incr)
    a.append(range(start, rnge.stop))

    return a

