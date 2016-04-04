from distutils.core import setup
from Cython.Build import cythonize
from setuptools import Extension
import numpy

ext_modules = [Extension(
    "skipgram",
    ["skipgram.pyx"])]

setup(
    ext_modules = ext_modules,
    include_dirs=[numpy.get_include()]
)
