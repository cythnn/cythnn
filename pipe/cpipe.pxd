from model.solution cimport Solution

cdef class CPipe:
    cdef:
        public object learner
        public object model
        Solution solution
        public int pipeid
