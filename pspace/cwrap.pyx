# distutils: language = c++
from cwrap cimport *

# Import numpy
cimport numpy as np
import numpy as np

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
    
    def basis(self, int k, np.ndarray[scalar, ndim=1, mode='c'] z):
        return self.ptr.basis(k, <scalar*> z.data)    
    def quadrature(self, int q):
        nparams = self.getNumParameters()
        cdef np.ndarray yq = None
        cdef np.ndarray zq = None
        yq = np.zeros(nparams, dtype=dtype)
        zq = np.zeros(nparams, dtype=dtype)
        self.ptr.quadrature(q, <scalar*> zq.data, <scalar*> yq.data)
        return zq, yq

    def getNumBasisTerms(self):        
        return self.ptr.getNumBasisTerms()
    def getNumParameters(self):
        return self.ptr.getNumParameters()
    def getNumQuadraturePoints(self):
        return self.ptr.getNumQuadraturePoints()

    ## def getBasisParamDeg(self, int k):
    ##     nparams = self.getNumParameters()
    ##     cdef np.ndarray degs = None
    ##     degs = np.zeros(nparams, dtype=int)
    ##     #self.ptr.getBasisParamDeg(k, <int*> degs.data)
    ##     return degs    
    ## def getBasisParamMaxDeg(self):
    ##     nparams = self.getNumParameters()
    ##     cdef np.ndarray pmax = None
    ##     pmax = np.zeros(nparams, dtype=int)
    ##     self.ptr.getBasisParamMaxDeg(<int*> pmax.data)
    ##     return pmax

    def initialize(self):
        self.ptr.initialize()
        return
    def initializeBasis(self, np.ndarray[int, ndim=1, mode='c'] pmax):
        self.ptr.initializeBasis(<int*> pmax.data)
        return
    def initializeQuadrature(self, np.ndarray[int, ndim=1, mode='c'] nqpts):
        self.ptr.initializeQuadrature(<int*> nqpts.data)
        return

    
