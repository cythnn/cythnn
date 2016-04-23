import cython
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("blas.cy",["blas/cy.pyx"]),
    Extension("matrix.cy",["matrix/cy.pyx"]),
    Extension("model.solution", ["model/solution.pyx"]),
    Extension("model.cpipe", ["model/cpipe.pyx"]),
    Extension("w2vContextWindows.cy", ["w2vContextWindows/cy.pyx"],),
    Extension("w2vHSoftmax.cy", ["w2vHSoftmax/cy.pyx"],),
    Extension("arch.SkipgramHS", ["arch/SkipgramHS.pyx"]),
    Extension("arch.SkipgramNS", ["arch/SkipgramNS.pyx"]),
    Extension("arch.CbowHS", ["arch/CbowHS.pyx"]),
    Extension("arch.CbowNS", ["arch/CbowNS.pyx"]),
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[np.get_include()]
)
