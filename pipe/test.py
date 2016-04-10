from pipe.cypipe_example import cypipe_example
from pipe.pypipe_example import pypipe_example

if __name__ == "__main__":
    # The first pipeline example starts with a pypipe, followed by two cypipes
    # The pypipe will simply use the feed method to pass results, the first cypipe
    # will use the binding to the cython process method to pass results
    m11 = pypipe_example(1, None)
    m12 = cypipe_example(1, None)
    m13 = cypipe_example(1, None)
    m13.bindTo(m12)
    m12.bindTo(m11)

    # The second pipeline example contains two cypipes
    # When using with model, the input is passed to the feed method of the first cypipe,
    # the remainder of the pipe shouls consist of only cypipes,
    # and the binding to the cython process method to pass results to successors
    m21 = cypipe_example(2, None)
    m22 = cypipe_example(2, None)
    m22.bindTo(m21)

    for i in range(10):
        # pipeline
        m11.feed(i)
        m21.feed(i)



