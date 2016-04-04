from distutils.core import setup
from Cython.Build import cythonize
import numpy
from setuptools import Extension

ext_modules = [
    Extension("tools.blas.blas",["tools/blas/blas.pyx"]),
    Extension("tools.nnmodel.model", ["tools/nnmodel/model.pyx"]),
    Extension("w2v.skipgram.skipgram", ["w2v/skipgram/skipgram.pyx"]),
    Extension("tools.hs.hs", ["tools/hs/hs.pyx"]),
    Extension("w2v.train.train", ["w2v/train/train.pyx"]),
    Extension("w2v.tr2.train", ["w2v/tr2/train.pyx"]),
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
