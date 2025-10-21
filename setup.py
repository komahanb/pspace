import os
import sys
import numpy

# Import distutils
from setuptools import setup
from distutils.core import Extension as Ext
from Cython.Build import cythonize

# Convert from local to absolute directories
def get_global_dir(files):
    root = os.path.abspath(os.path.dirname(__file__))
    new = []
    for f in files:
        new.append(os.path.join(root, f))
    return new

inc_dirs         = []
lib_dirs         = []
libs             = []
runtime_lib_dirs = []

# Add the numpy/mpi4py directories
inc_dirs.extend([numpy.get_include()])

# PSPACE
inc_dirs.extend(get_global_dir(['pspace']))
inc_dirs.extend(get_global_dir(['src/include']))
lib_dirs.extend(get_global_dir(['lib']))
runtime_lib_dirs.extend(lib_dirs)
libs.extend(['pspace'])

use_complex = (
    os.environ.get("PSPACE_COMPLEX") == "1"
    or any("PSPACE_USE_COMPLEX" in arg for arg in sys.argv)
)

define_macros = []
if use_complex:
    define_macros.append(('USE_COMPLEX', None))

compiler_directives = {"embedsignature": True, "binding": True}
compile_time_env = {"USE_COMPLEX": use_complex}

exts = []
for mod in ['PSPACE']:
    exts.append(Ext('pspace.%s'%(mod), sources=['pspace/%s.pyx'%(mod)],
                    include_dirs=inc_dirs,
                    libraries=libs,
                    library_dirs=lib_dirs,
                    runtime_library_dirs=runtime_lib_dirs,
                    define_macros=define_macros))

setup(name='pspace',
      version=1.0,
      description='Probabilistic space package for uncertainty quantification and optimization under uncertainty',
      author='Komahan Boopathy',
      author_email='komibuddy@gmail.com',
      ext_modules=cythonize(exts,
                            include_path=inc_dirs,
                            compiler_directives=compiler_directives,
                            compile_time_env=compile_time_env))
