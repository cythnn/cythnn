Deep Learning Text
===========
Extensible implementation to process text content in a deep learning network. This implementation requires Python 3+ and Cython to run. 

The standard implementation trains Word2Vec embeddings using skipgram and hierarchical softmax. The cython modules should be compiled on your system using comp.py in the project root. In the root, train_XXX.py is/are examples that learn embeddings from the words that appear in a flat text input file (e.g. data/text8). 

To learn a deep learning NN, a model must be configured, e.g.
```python
model(
	alpha=0.025, vectorsize=100,
	input=wordStreams("data/text8", byterange=byterange, parts=2),
	build=[ build_vocab, build_hs_tree, createW2V ],
	pipeline=[ convertWordIds, contextWindow, trainSkipgramHS ],
	mintf=5, cores=2, windowsize=5, iterations=1)
```

- input: can be configured as a list of word generators (in the example using wordStreams from tools.wordio).
- build: a list of functions that build prerequisites for learning the model:
 - build_vocab counts the corpus frequency of each word and stores it in the 'vocab' dictionary for other functions to use. The words are sorted by collection frequency (descending) and an end-of-sentence token </s> is added in first position. The words are then indexed on the position they hold to be later referred to by index number.
 - build_hierarchical_softmax builds a binary huffmann tree that is used in the output layer, and sets the output size to |vocab|-1
 - createW2V creates a solution list of 2 numpy arrays that represent the weight matrices w0 and w1 in a Word2Vec hierarchy. w1 = [|vocabulary|, |vectorsize|] and w2=[|output_layer_size|, |vector_size|] and seeds the matrices (w0 obtains random values, w1 zeros). After training, solution will contain the learned weight matrices. The input, output and hidden layer are not always required, but created on first use using model.getLayer(threadid, layer).

- pipeline: a list of modules (pipes) that extend pypipe (for full python) or cypipe (for cython). On construction, the modules are bound to its successor. When processing the input, per thread a separate instance of the pipeline is created and the first pipe is fed with one of the parts of the input, which passes its result onto the next pipe. Commonly, the last pipe in the pipeline is the learning module that updates the NN matrices, and the pipes before that perform some preparation step.

- mintf: filters the input keeping only words that occur at least mintf times in the corpus

- cores: number of threads to run in parallel

- windowsize: size of the context to be considered, e.g. 5 means a maximum of 5 words to the left and right. In accordance with the original implementation the actual window size per word is uniformly sampled between 1 and windowsize and bound by sentence boundaries.

- iterations: the number of times to pass over the corpus for training

The model is projected to a C-version of the model that shares the memory for the solution. Methods on modelc can be used to obtain a reference to the solutions shared weight matrices or thread-safe layers. 

Pipes
=====

The pipeline can be constructed of configurable modules. A valid pipeline may start with pypipes written in Python (e.g. convertWordIds) and at some point can cross to cypipes written in Cython (e.g. ContextWindow, trainSkipgramHS). Once the crossover to cython is made, the pipeline cannot return to python. Also, to allow multithreading, the GIL must be released by specifically using 'with nogil:', it will not work by just calling a nogil function.

Each pipe should override the default __init__ if it needs to do a setup prior to processing. During initialization, it can access the python model which can be customized with additional parameters and used to set instance variables. For efficient processing of cypipes under nogil, the python model is no longer accessible. 

To allow pipes to be bound to their successor, cypipes should implement 'bindTo' in which they pass a reference to their processing method to their predecessor, and 'bind' in which they register their successor and its processing method. It is recommended to cast the void* reference to the processing method to it proper type during initialization, and cast a readable error if the types fail.   

In the example, the following pipes are used:
- ConvertWordIds (ConvertWordIds.py): transforms the stream of words (strings) to an array of word index numbers by lookup in vocab
- ContextWindow (w2vContextWindows.cy): adds context boundaries to every word position, by sampling a window size from a uniform distribution and taking sentence boundaries and file-part cutoffs into account. 
- trainSkipgramHS (w2vSkipgramHS.cy): learns the embeddings using Skipgram and Hierarchical Softmax

he data folder contains a few data samples, of which text8 is a standard collection consisting of a preprocessed for 100M of Wikipedia that can be used to benchmark against other implementations. A learned model can be saved in binary form, you can use with the original evaluation tool written in C (http://http://word2vec.googlecode.com/svn/trunk/) or the evaluation tool in Gensim for Python.

./compute-accuracy saved_model.bin <questions-words.txt 
