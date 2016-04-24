import cython
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("tools.blas",["tools/blas.pyx"]),
    Extension("tools.matrix",["tools/matrix.pyx"]),
    Extension("model.solution", ["model/solution.pyx"]),
    Extension("pipe.cpipe", ["pipe/cpipe.pyx"]),
    Extension("pipe.ContextWindows", ["pipe/ContextWindows.pyx"],),
    Extension("pipe.DownSample", ["pipe/DownSample.pyx"],),
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
