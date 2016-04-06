import cython

from model.model cimport model_c, cINT, cREAL
from numpy cimport ndarray

cdef void trainW2V(int threadid, int pipelineindex, model_c model, cINT *samples,
                    int length, float alpha) nogil

