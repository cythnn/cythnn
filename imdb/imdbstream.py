import re
from tools.recordio import RecordStream

# reads tsv records that are in a flat text file, separated by newlines.
# chunks contain all record whose recordseparator (usually a newline) is within the given range,
# and read past the upper range boundary to include that record. Therefore, a chunk that
# does not start at the beginning ignores all bytes until the first recordseparator.

class ImdbStream:
    def __init__(self, model, input, inputrange=None):
        self.inputrange=inputrange
        self.recordstream = RecordStream(input=input, inputrange=inputrange)
        self.wordsplitter = re.compile(r'\s')

    def __iter__(self):
        self.id = ""
        for record in self.recordstream:
            if len(record) < 6:
                print(record)
            else:
                self.imdbid = int(record[0])
                self.userid = int(record[1])             # not used right now
                self.rating = int(record[2])
                self.useful = int(record[3])
                self.usefultotal = int(record[4])
                reviewline = record[5]
                yield "#" + record[0]
                for word in self.wordsplitter.split(reviewline):
                    yield word
                #yield '</s>'                    # we don't need eol, since the imdbid will separate the sentences





