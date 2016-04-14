from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension("blas.cy",["blas/cy.pyx"]),
    Extension("tokyo",["tokyo/tokyo.pyx"]),
    Extension("matrix.cy",["matrix/cy.pyx"]),
    Extension("model.cy", ["model/cy.pyx"]),
    Extension("pipe.cy", ["pipe/cy.pyx"]),
    Extension("pipe.cy_example", ["pipe/cy_example.pyx"]),
    Extension("w2vContextWindows.cy", ["w2vContextWindows/cy.pyx"],),
    Extension("w2vHSoftmax.cy", ["w2vHSoftmax/cy.pyx"],),
    Extension("w2vSkipgramHS.cy", ["w2vSkipgramHS/cy.pyx"]),
    Extension("w2vSkipgramNS.cy", ["w2vSkipgramNS/cy.pyx"]),
    Extension("w2vCbowHS.cy", ["w2vCbowHS/cy.pyx"]),
    Extension("w2vCbowNS.cy", ["w2vCbowNS/cy.pyx"]),
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
