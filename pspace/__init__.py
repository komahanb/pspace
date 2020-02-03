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

    rel_inc_dirs = ['cpp']

    inc_dirs = []
    for path in rel_inc_dirs:
    	inc_dirs.append(os.path.join(root_path, path))

    return inc_dirs

def get_libraries():
    '''
    Get the library directories
    '''
    root_path, tail = os.path.split(os.path.abspath(os.path.dirname(__file__)))

    rel_lib_dirs = ['cpp']
    libs = ['pspace']
    lib_dirs = []
    for path in rel_lib_dirs:
    	lib_dirs.append(os.path.join(root_path, path))

    return lib_dirs, libs
