# A pypipe is a lightweight pure python pipeline class, that performs a processing task on the input and
# passes on their results to the next module in the pipeline.
# A pypipe extension must implement a feed(self, input) method to proccess the received input and pass it results
# to its successor's feed method
class pypipe:
    def __init__(self, threadid, model):
        self.model = model
        self.threadid = threadid

    # for compatibility with cypipe, binds a cypipe to its successor in the pipeline
    def bind(self, successor):

        self.successor = successor

    def bindTo(self, predecessor):
        predecessor.bind(self)

    def feed(self, input):
        raise NotImplementedError("A pypipe must implement the feed() method.")
