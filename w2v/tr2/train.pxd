import cython

from tools.hs.hs cimport trainerWrapper
from tools.nnmodel.model cimport model_c, cINT, cREAL
from numpy cimport ndarray

cdef void train_c(int threadid, cREAL *hiddenlayer, int vectorsize,
                      cREAL *w0, cREAL *w1, cINT *samples,
                      int length, cREAL *exptable, float alpha) nogil
