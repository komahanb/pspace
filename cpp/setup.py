##from distutils.core import setup, Extension
##from Cython.Build import cythonize
##
##setup(ext_modules = cythonize(Extension(
##           "rect",                                # the extesion name
##           sources=["rect.pyx", "Rectangle.cpp"], # the Cython source and
##                                                  # additional C++ source files
##           language="c++",                        # generate and compile C++ code
##      )))
##
##

from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "corthogonal_polynomials",                                # the extesion name
           sources=["corthogonal_polynomials.pyx", "OrthogonalPolynomials.cpp"], # the Cython source and                                                  # additional C++ source files
           language="c++",                        # generate and compile C++ code
      )))
