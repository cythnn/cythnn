Deep Learning Text
===========
Extensible implementation to process text content in a deep learning network. This implementation requires Python 3+ and Cython to run. 

The standard implementation trains Word2Vec embeddings using skipgram and hierarchical softmax. The cython modules should be compiled on your system using setup.py from the project root (python3 setup.py build_ext --inplace). In the root, train_XXX.py are examples that learn embeddings from the words that appear in a flat text input file (e.g. data/text8). 

To learn a deep learning architecture, a model must be configured, e.g.
```python
model(
	alpha=0.025, vectorsize=100,
	input="data/text8",
	inputrange=None,
	build=[ build_vocab ],
	pipeline=[ createW2VInputTasks, convertWordIds, DownSample, contextWindow, trainSkipgramHS ],
	mintf=5, cores=2, windowsize=5, iterations=1
	iterations=1, downsample=0)
```

- input: a single filename 
- build: a list of functions that build prerequisites for learning the model:
 - build_vocab counts the corpus frequency of each word and stores it in the 'vocab' dictionary for other functions to use. The words are sorted by collection frequency (descending) and an end-of-sentence token </s> is added in first position. The words are then indexed on the position they hold to be later referred to by index number.
 - Pipeline modules that have requirements to be built can extend a build() method in which they call the function, such as:
  - hsoftmax to build a Huffmann tree, and sets the output layer size to |vocab| - 1. 
  - createW2V creates a solution space consisting of 2 numpy arrays that represent the weight matrices w0 and w1 in a Word2Vec hierarchy. w1 = [|vocabulary|, |vectorsize|] and w2=[|output_layer_size|, |vector_size|] and seeds the matrices (w0 obtains random values, w1 zeros). After training, solution will contain the learned weight matrices. The input, output and hidden layer are not always required, but created on first use using model.getLayer(threadid, layer).

- pipeline: a list of modules (pipes) that extend Pipe (for full python) or CPipe (for cython). A Pipe has a feed() method that processes one task, and results in a (set of) new task for the next Pipe, or updates to the solution for a learning module at the end of the pipeline. More on pipes below.

- mintf: filters the input keeping only words that occur at least mintf times in the corpus

- cores/threads: number of cores/threads, when both are specified, the number of cores is used to build the prerequisites and the number of threads for learning.

- windowsize: size of the context to be considered, e.g. 5 means a maximum of 5 words to the left and right. In accordance with the original implementation the actual window size per word is uniformly sampled between 1 and windowsize and bound by sentence boundaries.

- iterations: the number of times to pass over the corpus for training

During learning, the model is accompanied by a Solution object, which is accessible from Cython in nogil (for fast parallel processing).

Pipes
=====

The pipeline can be constructed of configurable modules. For convenience, pipes may be written in Python when they are not critical to the processing speed (e.g. convertWordIds) or Cython when speed is of essence (e.g. ContextWindow, trainSkipgramHS). True parallel processing only occurs when running Cython code in nogil mode (note: the GIL must be released by specifically using 'with nogil:', it will not work by just calling a nogil function).

Each pipe can override the default __init__ if it needs to do a setup prior to processing. During initialization, it can access the model which is used to pass any (custom) configuration. A pipe must also override the feed() method, which is used to pass a task for processing. Typically, all pipes but the last push their results as new tasks to the learner for processing by the next pipe in the pipeline. Optionally, a pipe can implement a build() method, which is triggered before its __init__() to build prerequisites for processing, and modify the model. Lastly, the transform method is called after initialization, which can be used by a Pipe to remove itself from the pipeline by returning None (e.g. DownSample removes itself when the model contains downsample=0) or replace itself by another module (e.g. SkipgramHS replaces itself by SkipgramHScached when caching is turned on).  

To allow pipes to be bound to their successor, cypipes should implement 'bindTo' in which they pass a reference to their processing method to their predecessor, and 'bind' in which they register their successor and its processing method. It is recommended to cast the void* reference to the processing method to it proper type during initialization, and cast a readable error if the types fail.   

In the example, the following pipes are used:
- CreateW2VInputTasks: splits the input file into tasks that can be processed in parallel.
- ConvertWordIds (ConvertWordIds.py): transforms the stream of words (strings) to an array of word index numbers by lookup in vocab
- DownSample.py: reduces the occurrences of frequently occurring terms using the downsample parameter
- ContextWindow (w2vContextWindows.cy): adds context boundaries to every word position, by sampling a window size from a uniform distribution and taking sentence boundaries and file-part cutoffs into account. 
- trainSkipgramHS (w2vSkipgramHS.cy): learns the embeddings using Skipgram and Hierarchical Softmax

he data folder contains a few data samples, of which text8 is a standard collection consisting of a preprocessed for 100M of Wikipedia that can be used to benchmark against other implementations. A learned model can be saved in binary form, you can use with the original evaluation tool written in C (http://http://word2vec.googlecode.com/svn/trunk/) or the evaluation tool in Gensim for Python.

./compute-accuracy saved_model.bin <questions-words.txt 
