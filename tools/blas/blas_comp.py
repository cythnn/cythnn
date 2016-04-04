from distutils.core import setup
from Cython.Build import cythonize
import numpy
from setuptools import Extension

ext_modules = [Extension(
    "blas",
    ["blas.pyx"])]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)

#setup(ext_modules = cythonize(ext_modules))