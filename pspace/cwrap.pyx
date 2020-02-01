# distutils: language = c++
from cwrap cimport *
import numpy as np

# Include the definitions
include "PspaceDefs.pxi"

cdef class PyAbstractParameter:
    cdef AbstractParameter *ptr
    def __cinit__(self):
        self.ptr = NULL
        return
    def basis(self, scalar z, int d):
        return self.ptr.basis(z,d)

cdef class PyParameterFactory:
    cdef ParameterFactory *ptr
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
    cdef ParameterContainer *ptr
    def __cinit__(self, int basis_type=0, int quadrature_type=0):
        self.ptr = new ParameterContainer(basis_type, quadrature_type)
        return
    def addParameter(self, PyAbstractParameter param):
        self.ptr.addParameter(param.ptr)
        return
    def getNumBasisTerms(self):        
        return self.ptr.getNumBasisTerms()
    def getNumParameters(self):
        return self.ptr.getNumParameters()
    def getNumQuadraturePoints(self):
        return self.ptr.getNumQuadraturePoints()
    def initialize(self):
        self.ptr.initialize()
        return
    
    
## cdef class PyNormalParameter(PyAbstractParameter):
##     cdef NormalParameter *dptr
##     def __cinit__(self, int pid, scalar mu, scalar sigma):
##         self.dptr = new NormalParameter(pid, mu, sigma)
##         self.ptr = self.dptr
##         return
