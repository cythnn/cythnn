from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension("blas.blas",["blas/blas.pyx"]),
    Extension("model.model", ["model/model.pyx"]),
    Extension("hs.hs", ["hs/hs.pyx"],),
    Extension("w2vTrainer.train", ["w2vTrainer/train.pyx"]),
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
