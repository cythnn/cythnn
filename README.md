Deep Learning Text
===========
Extensible implementation to process text content in a deep learning network. This implementation requires Python 3+ and Cython to run. 

The standard implementation trains Word2Vec embeddings using skipgram and hierarchical softmax. The cython modules should be compiled on your system using comp.py in the project root. In the root, w2vf.py is an example that learns embeddings from the words that appear in a flat text input file (e.g. data/text8). The input can be configured as a list of word generators (in the example wordStreams provides these). The model is prepared by calling the functions in the build array, which add requirements to the model:
- build_vocab counts the corpus frequency of each word and stores it in the 'vocab' dictionary for other functions to use. The words are sorted by collection frequency (descending) and an end-of-sentence token </s> is added in first position. The words are then indexed on the position they hold to be later referred to by index number.
- build_hierarchical_softmax builds a binary huffmann tree that is used in the output layer, and sets the output size to |vocab|-1
- createW2V creates a solution list of 2 numpy arrays that represent the weight matrices w0 and w1 in a Word2Vec hierarchy. w1 = [|vocabulary|, |vectorsize|] and w2=[|output_layer_size|, |vector_size|] and seeds the matrices (w0 obtains random values, w1 zeros). After training, solution will contain the learned weight matrices. The input, output and hidden layer are not always required, but created on first use using model.getLayer(threadid, layer).

The model is projected to a C-version of the model that shares the memory for the solution. Since in C/Cython all class attributes must be specified at compile, necessary extension should be made to the code.

Processing is performed in a pipeline, starting with one or more Python modules by passing (threadid, model, feed) to each module sequentially. Initially, feed will consist of a word generator, the output of the first module will be used to feed the next. The modules must be able to consume whatever the previous module has returned as result. Then a crossover is made from Python to Cython modules, which allows for efficient multithreaded learning of the actual model through fast C-code combined with blas functions. The crossover to C-code must be done through a Python module that converts the last feed of Python objects into C-parameters. Consecutive C-modules must have the exact same parameters as the output of its predecessor:
- readWordIds transforms the stream of words to an array of word index numbes using vocab
- processhs converts the feed to the Cython processhs2 module. Processhs2 then converts the array of word-ids into an array of observed triples (target_word, inner node from the huffman tree, expected value (0 indicates left turn, 1 indicates right turn). Processhs2 fills an array of observations to be passed down the pipeline, and keeps rack of the number of words that have been trained and updates the learning rate alpha = perc words learned * initial_alpha
- addTrainW2V is a python function that adds the Cython function trainW2V to the cython pipeline of the model. Each cython model receives a threadid, the C-representation of the model, and additional arguments for processing. Since in C parameters are fixed, successive modules should accept the arguments provided by its predecessor. No type checking is done, but a runtime exception is to be expected when they do not.

he data folder contains a few data samples, of which text8 is a standard collection consisting of a preprocessed for 100M of Wikipedia that can be used to benchmark against other implementations. A learned model can be saved in binary form, you can use with the original evaluation tool written in C (http://http://word2vec.googlecode.com/svn/trunk/) or the evaluation tool in Gensim for Python.

./compute-accuracy saved_model.bin <questions-words.txt 
