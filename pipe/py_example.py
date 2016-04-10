from pipe.pypipe import pypipe

# A pypipe is a lightweight pure python pipeline class, that performs a processing task on the input and
# passes on their results to the next module in the pipeline.
# A pypipe extension must implement a feed(self, input) method to proccess the received input and pass it results
# to its successor's feed method
class pypipe_example(pypipe):
    # by default, pypipe.__init__ registers self.threadid and self.model to provide access to the model specifications
    # optionally the constructor can be overridden to initialize the module according to the model
    def __init__(self, threadid, model):
        pypipe.__init__(self, threadid, model)

    def feed(self, input):
        print("pypipe_example", self.threadid, input)
        self.successor.feed(input + 1)

