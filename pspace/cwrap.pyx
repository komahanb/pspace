# distutils: language = c++
from cwrap cimport NormalParameter
import numpy as np

# Include the definitions
include "PspaceDefs.pxi"

cdef class PyNormalParameter:
    cdef NormalParameter *ptr
    def __cinit__(self, int pid, scalar mu, scalar sigma):
        self.ptr = new NormalParameter(pid, mu, sigma)
        return
