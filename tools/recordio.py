import os, math, re

# reads tsv records that are in a flat text file, separated by newlines.
# chunks contain all record whose recordseparator (usually a newline) is within the given range,
# and read past the upper range boundary to include that record. Therefore, a chunk that
# does not start at the beginning ignores all bytes until the first recordseparator.
from tools.wordio import size

class RecordStream:
    class recorditerator:
        def __init__(self, chunk, input, recordseparator):
            self.recordseparator = re.compile(recordseparator)
            self.chunk = chunk
            self.input = input

        def __iter__(self):
            with open(self.input, "r") as f:
                offset = self.chunk.start
                if offset > 0:
                    f.seek(offset)
                lastpos = None
                buffer = ""
                while lastpos is None or lastpos.start() + offset < self.chunk.stop:
                    if lastpos is not None:
                        buffer = buffer[lastpos.end():]
                        offset += lastpos.end()
                    newbuf = f.read(1000000)
                    if not newbuf:
                        if len(buffer) > 0:
                            yield buffer
                        break
                    else:
                        buffer += newbuf
                        for position in self.recordseparator.finditer(buffer):
                            if lastpos is not None:
                                yield buffer[lastpos.end():position.start()]
                            lastpos = position
                            if offset + lastpos.start() >= self.chunk.stop:
                                break

    def __init__(self, inputrange=None, input=None, recordseparator=r'\n', fieldseparator=r'\t'):
        self.input = input
        if input and inputrange is None:
            self.inputrange = range(0, size(input))
        else:
            self.inputrange = inputrange
        self.recordseparator = recordseparator
        self.fieldseparator = re.compile(fieldseparator)

    def __iter__(self):
        for record in self.recorditerator(self.inputrange, self.input, self.recordseparator):
            recs = self.fieldseparator.split(record)
            if len(recs) == 6:
                yield recs


