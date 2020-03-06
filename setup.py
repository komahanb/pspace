import os
from subprocess import check_output
import sys

# Numpy/mpi4py must be installed prior to installing TACS
import numpy

# Import distutils
from setuptools import setup
from distutils.core import Extension as Ext
from Cython.Build import cythonize

# Convert from local to absolute directories
def get_global_dir(files):
    tacs_root = os.path.abspath(os.path.dirname(__file__))
    new = []
    for f in files:
        new.append(os.path.join(tacs_root, f))
    return new

inc_dirs = []
lib_dirs = []
libs = []

# Add the numpy/mpi4py directories
inc_dirs.extend([numpy.get_include()])

# PSPACE
inc_dirs.extend(get_global_dir(['cpp']))
lib_dirs.extend(get_global_dir(['cpp']))

# The provide where the run time libraries are present
runtime_lib_dirs = []
runtime_lib_dirs.extend(lib_dirs)

libs.extend(['pspace'])

exts = []
for mod in ['PSPACE']:
    exts.append(Ext('pspace.%s'%(mod), sources=['pspace/%s.pyx'%(mod)],
                    include_dirs=inc_dirs,
                    libraries=libs,
                    library_dirs=lib_dirs,
                    runtime_library_dirs=runtime_lib_dirs,
                    cython_directives={"embedsignature": True, "binding": True}))

setup(name='pspace',
      version=1.0,
      description='Probabilistic space package for uncertainty quantification and optimization under uncertainty',
      author='Komahan Boopathy',
      author_email='komibuddy@gmail.com',
      ext_modules=cythonize(exts, include_path=inc_dirs))
