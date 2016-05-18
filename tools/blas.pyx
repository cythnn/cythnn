import scipy.linalg.blas as fblas

# from gensim
# bind scipy blas functions, for very fast vector computations

cdef:
    # scopy: y = x
    scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)

    # saxpy: y = y + a * x
    saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)

    # sdot: (single precision float) = x * y
    sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)

    # snrm2: = sqrt( x * x )
    snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)

    # sscal: x = a * x
    sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer)

    # sswap: x,y = y,x
    sswap_ptr sswap=<sswap_ptr>PyCObject_AsVoidPtr(fblas.sswap._cpointer)
