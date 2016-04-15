from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_params = {}
ext_params['include_dirs'] = [
    '/usr/include',
    '/System/Library/Frameworks/vecLib.framework/Versions/A/Headers',
    np.get_include()]
ext_params['extra_compile_args'] = ["-O2"]
ext_params['extra_link_args'] = ["-Wl,-O1", "-Wl,--as-needed"]  # TODO: as-needed

ext_params['libraries'] = ['blas']  # TODO: detect library name.
    # Candidates: blas, cblas, lapack, lapack_atlas, atlas
    # On OSX, blas points to the Accelerate framework's ATLAS library.
ext_params['library_dirs'] = ['/usr/lib']  # needed by OSX, perhaps

ext_modules = [
    #Extension("blas.cy",["blas/cy.pyx"]),
    Extension("tokyo",["tokyo/tokyo.pyx"], **ext_params),
 #   Extension("matrix.cy",["matrix/cy.pyx"]),
 #   Extension("model.cy", ["model/cy.pyx"]),
 #   Extension("pipe.cy", ["pipe/cy.pyx"]),
 #   Extension("pipe.cy_example", ["pipe/cy_example.pyx"]),
 #   Extension("w2vContextWindows.cy", ["w2vContextWindows/cy.pyx"],),
 #   Extension("w2vHSoftmax.cy", ["w2vHSoftmax/cy.pyx"],),
 #   Extension("w2vSkipgramHS.cy", ["w2vSkipgramHS/cy.pyx"]),
 #   Extension("w2vSkipgramNS.cy", ["w2vSkipgramNS/cy.pyx"]),
 #   Extension("w2vCbowHS.cy", ["w2vCbowHS/cy.pyx"]),
 #   Extension("w2vCbowNS.cy", ["w2vCbowNS/cy.pyx"]),
]

setup(
    ext_modules = cythonize(ext_modules)
)
