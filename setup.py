import os
from subprocess import check_output
import sys

# Numpy/mpi4py must be installed prior to installing TACS
import numpy
import mpi4py
import tacs
import tmr

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

def get_mpi_flags():
    # Split the output from the mpicxx command
    args = check_output(['mpicxx', '-show']).decode('utf-8').split()

    # Determine whether the output is an include/link/lib command
    inc_dirs, lib_dirs, libs = [], [], []
    for flag in args:
        if flag[:2] == '-I':
            inc_dirs.append(flag[2:])
        elif flag[:2] == '-L':
            lib_dirs.append(flag[2:])
        elif flag[:2] == '-l':
            libs.append(flag[2:])

    return inc_dirs, lib_dirs, libs

inc_dirs, lib_dirs, libs = get_mpi_flags()

# Add the numpy/mpi4py directories
inc_dirs.extend([numpy.get_include(), mpi4py.get_include()])

# PSPACE
inc_dirs.extend(get_global_dir(['cpp']))
lib_dirs.extend(get_global_dir(['cpp']))

# TACS
inc_dirs.extend(tacs.get_include())
inc_dirs.extend(tacs.get_cython_include())
lib_dirs.extend(tacs.get_libraries()[0])

# STACS
inc_dirs.extend(get_global_dir(["examples/stacs/cpp"]))
lib_dirs.extend(get_global_dir(["examples/stacs/cpp"]))

# TMR
inc_dirs.extend(tmr.get_include())
inc_dirs.extend(tmr.get_cython_include())
lib_dirs.extend(tmr.get_libraries()[0])

# The provide where the run time libraries are present
runtime_lib_dirs = [] 
runtime_lib_dirs.extend(lib_dirs)

libs.extend(['pspace', 'tacs', 'stacs', 'tmr'])

print(inc_dirs)
print(libs)
print(lib_dirs)
print(runtime_lib_dirs)

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
      description='Parallel finite-element analysis package',
      author='Komahan Boopathy',
      author_email='komibuddy@gmail.com',
      ext_modules=cythonize(exts, include_path=inc_dirs))
