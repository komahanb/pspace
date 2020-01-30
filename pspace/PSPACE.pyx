# distutils: language = c++

from PSPACE cimport AbstractParameter

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class PyAbstractParameter:
    cdef AbstractParameter ptr  # Hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.ptr = AbstractParameter()
