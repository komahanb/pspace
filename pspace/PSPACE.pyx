# distutils: language = c++
# cython: language_level=3
from libc.stddef cimport size_t
from libcpp.complex cimport complex

cdef extern from "<complex>" namespace "std":
    double real(complex[double] z)
    double imag(complex[double] z)

from cpython.complex cimport PyComplex_FromDoubles

from pspace.PSPACE cimport *

import numpy as np
numpy = np

include "_config.pxi"

cdef object PSPACE_NPY_SCALAR

IF USE_COMPLEX:
    cdef inline object _scalar_to_python(scalar value):
        cdef double _real = real(value)
        cdef double _imag = imag(value)
        return PyComplex_FromDoubles(_real, _imag)
ELSE:
    cdef inline object _scalar_to_python(double value):
        return value

if USE_COMPLEX:
    PSPACE_NPY_SCALAR = np.complexfloating
else:
    PSPACE_NPY_SCALAR = np.floating

cdef class PyAbstractParameter:
    def __cinit__(self):
        self.ptr = NULL
        return
    def basis(self, scalar z, int d):
        cdef scalar result = self.ptr.basis(z,d)
        return _scalar_to_python(result)

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
        dtype_local = _np.complex128 if USE_COMPLEX else _np.double
        z_arr, z_ptr_obj = _ptr_utils.ensure_scalar_pointer(z, dtype=dtype_local)
        cdef size_t z_ptr = z_ptr_obj
        cdef scalar result = self.ptr.basis(k, <scalar*> z_ptr)
        return _scalar_to_python(result)
    def quadrature(self, int q, bint with_weight=False):
        import numpy as _np
        import pspace._pointer_utils as _ptr_utils
        nparams = self.getNumParameters()
        dtype_local = _np.complex128 if USE_COMPLEX else _np.double
        zq, z_ptr_obj = _ptr_utils.ensure_scalar_pointer(_np.zeros(nparams, dtype=dtype_local), dtype=dtype_local)
        yq, y_ptr_obj = _ptr_utils.ensure_scalar_pointer(_np.zeros(nparams, dtype=dtype_local), dtype=dtype_local)
        cdef size_t z_ptr = z_ptr_obj
        cdef size_t y_ptr = y_ptr_obj
        cdef scalar wq = self.ptr.quadrature(q, <scalar*> z_ptr, <scalar*> y_ptr)
        if with_weight:
            return _scalar_to_python(wq), zq, yq
        return zq, yq

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
