import os

def get_cython_include():
    '''
    Get the include directory for the Cython .pxd files in PSPACE
    '''
    return [os.path.abspath(os.path.dirname(__file__))]

def get_include():
    '''
    Get the include directory for the Cython .pxd files in PSPACE
    '''
    root_path, tail = os.path.split(os.path.abspath(os.path.dirname(__file__)))

    rel_inc_dirs = ['src/include']

    inc_dirs = []
    for path in rel_inc_dirs:
    	inc_dirs.append(os.path.join(root_path, path))

    return inc_dirs

def get_libraries():
    '''
    Get the library directories
    '''
    root_path, tail = os.path.split(os.path.abspath(os.path.dirname(__file__)))

    rel_lib_dirs = ['lib']
    libs = ['pspace']
    lib_dirs = []
    for path in rel_lib_dirs:
    	lib_dirs.append(os.path.join(root_path, path))

    return lib_dirs, libs

# Optional re-exports for convenience
from .core import CoordinateSystem as NumericCoordinateSystem  # noqa: E402
from .symbolic import CoordinateSystem as SymbolicCoordinateSystem  # noqa: E402
from .analytic import CoordinateSystem as AnalyticCoordinateSystem  # noqa: E402
from .profile import CoordinateSystem as ProfileCoordinateSystem  # noqa: E402
from .validate import CoordinateSystem as ValidateCoordinateSystem  # noqa: E402
from .verify import CoordinateSystem as VerifyCoordinateSystem  # noqa: E402
from .sparsity import CoordinateSystem as SparsityCoordinateSystem  # noqa: E402
from .parallel import ParallelCoordinateSystem  # noqa: E402
