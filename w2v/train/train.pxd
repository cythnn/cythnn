import cython
from tools.nnmodel.model cimport *
from numpy cimport ndarray

cdef void train_c(int threadid, model_c m, ndarray samples, float alpha)

