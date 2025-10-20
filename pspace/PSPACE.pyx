# distutils: language = c++
# cython: language_level=3
from libc.stddef cimport size_t

from PSPACE cimport *

import numpy as np
numpy = np

include "PspaceDefs.pxi"

cdef class PyAbstractParameter:
    def __cinit__(self):
        self.ptr = NULL
        return
    def basis(self, scalar z, int d):
        return self.ptr.basis(z,d)

cdef class PyParameterFactory:
    def __cinit__(self):
        self.ptr = new ParameterFactory()
        return
    def createNormalParameter(self, scalar mu, scalar sigma, int dmax):
        cdef PyAbstractParameter pyparam = PyAbstractParameter()
        pyparam.ptr = self.ptr.createNormalParameter(mu, sigma, dmax)
        return pyparam
    def createUniformParameter(self, scalar a, scalar b, int dmax):
        cdef PyAbstractParameter pyparam = PyAbstractParameter()
        pyparam.ptr = self.ptr.createUniformParameter(a, b, dmax)
        return pyparam
    def createExponentialParameter(self, scalar mu, scalar beta, int dmax):
        cdef PyAbstractParameter pyparam = PyAbstractParameter()
        pyparam.ptr = self.ptr.createExponentialParameter(mu, beta, dmax)
        return pyparam

cdef class PyParameterContainer:
    def __cinit__(self, int basis_type=0, int quadrature_type=0):
        self.ptr = new ParameterContainer(basis_type, quadrature_type)
        return

    def addParameter(self, PyAbstractParameter param):
       self.ptr.addParameter(param.ptr)
       return

    def basis(self, int k, object z):
        import numpy as _np
        import pspace._pointer_utils as _ptr_utils
        z_arr, z_ptr_obj = _ptr_utils.ensure_scalar_pointer(z, dtype=_np.double)
        cdef size_t z_ptr = z_ptr_obj
        return self.ptr.basis(k, <scalar*> z_ptr)
    def quadrature(self, int q):
        import numpy as _np
        import pspace._pointer_utils as _ptr_utils
        nparams = self.getNumParameters()
        zq, z_ptr_obj = _ptr_utils.ensure_scalar_pointer(_np.zeros(nparams, dtype=_np.double), dtype=_np.double)
        yq, y_ptr_obj = _ptr_utils.ensure_scalar_pointer(_np.zeros(nparams, dtype=_np.double), dtype=_np.double)
        cdef size_t z_ptr = z_ptr_obj
        cdef size_t y_ptr = y_ptr_obj
        wq = self.ptr.quadrature(q, <scalar*> z_ptr, <scalar*> y_ptr)
        return wq, zq, yq

    def getNumBasisTerms(self):
        return self.ptr.getNumBasisTerms()
    def getNumParameters(self):
        return self.ptr.getNumParameters()
    def getNumQuadraturePoints(self):
        return self.ptr.getNumQuadraturePoints()

    def getBasisParamDeg(self, int k):
        import numpy as _np
        import pspace._pointer_utils as _ptr_utils
        nparams = self.getNumParameters()
        degs, deg_ptr_obj = _ptr_utils.ensure_int_pointer(_np.zeros(nparams, dtype=_np.intc), dtype=_np.intc)
        cdef size_t deg_ptr = deg_ptr_obj
        self.ptr.getBasisParamDeg(k, <int*> deg_ptr)
        return degs
    def getBasisParamMaxDeg(self):
        import numpy as _np
        import pspace._pointer_utils as _ptr_utils
        nparams = self.getNumParameters()
        pmax, pmax_ptr_obj = _ptr_utils.ensure_int_pointer(_np.zeros(nparams, dtype=_np.intc), dtype=_np.intc)
        cdef size_t pmax_ptr = pmax_ptr_obj
        self.ptr.getBasisParamMaxDeg(<int*> pmax_ptr)
        return pmax

    def initialize(self):
        self.ptr.initialize()
        return
    def initializeBasis(self, object pmax):
        import numpy as _np
        import pspace._pointer_utils as _ptr_utils
        pmax_arr, pmax_ptr_obj = _ptr_utils.ensure_int_pointer(pmax, dtype=_np.intc)
        cdef size_t pmax_ptr = pmax_ptr_obj
        self.ptr.initializeBasis(<int*> pmax_ptr)
        return
    def initializeQuadrature(self, object nqpts):
        import numpy as _np
        import pspace._pointer_utils as _ptr_utils
        nqpts_arr, nq_ptr_obj = _ptr_utils.ensure_int_pointer(nqpts, dtype=_np.intc)
        cdef size_t nq_ptr = nq_ptr_obj
        self.ptr.initializeQuadrature(<int*> nq_ptr)
        return
