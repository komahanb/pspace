# distutils: language = c++
from PSPACE cimport *

# Import numpy
cimport numpy as np
import numpy as np
np.import_array()

# Import C methods for python
from cpython cimport PyObject, Py_INCREF, Py_DECREF

include "PspaceDefs.pxi"
include "TacsDefs.pxi"

cdef inplace_array_1d(int nptype, int dim1, void *data_ptr):
    '''Return a numpy version of the array'''
    # Set the shape of the array
    cdef int size = 1
    cdef np.npy_intp shape[1]
    cdef np.ndarray ndarray

    # Set the first entry of the shape array
    shape[0] = <np.npy_intp>dim1

    # Create the array itself - Note that this function will not
    # delete the data once the ndarray goes out of scope
    ndarray = np.PyArray_SimpleNewFromData(size, shape, nptype, data_ptr)

    return ndarray

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

cdef void updateCB(TACSElement *elem, TacsScalar *yvals, void *pyptr):
    _yvals = inplace_array_1d(TACS_NPY_SCALAR, 5, <void*> yvals)
    (<object>pyptr).update(_yvals)
    return

cdef class PySMD(Element):
    cdef SMD *smd
    def __cinit__(self, TacsScalar m, TacsScalar c, TacsScalar k):
        self.smd = new SMD(m, c, k)
        self.ptr = self.smd
        self.ptr.incref()
        Py_INCREF(self)
    def __dealloc__(self):
        if self.ptr:
            self.ptr.decref()
            Py_DECREF(self)
    def setMass(self, TacsScalar m):
        return self.smd.setMass(m)
    def setStiffness(self, TacsScalar k):
        return self.smd.setStiffness(k)
    def setDamping(self, TacsScalar c):
        return self.smd.setDamping(c)

cdef class PyStochasticElement(Element):
    cdef TACSStochasticElement *sptr
    def __cinit__(self, Element elem,
                  PyParameterContainer pc,
                  update):
        self.sptr = new TACSStochasticElement(elem.ptr, pc.ptr, &updateCB)
        self.sptr.incref()        
        self.sptr.setPythonCallback(<PyObject*>update)
        self.ptr = self.sptr
        Py_INCREF(update)
        Py_INCREF(self)
    def __dealloc__(self):        
        if self.sptr:
            self.sptr.decref()
            Py_DECREF(self)
    def getDeterministicElement(self):
        delem = Element()
        delem.ptr = self.sptr.getDeterministicElement() 
        delem.ptr.incref()
        return delem
    def updateElement(self, Element elem, np.ndarray[TacsScalar, ndim=1, mode='c'] vals):
        self.sptr.updateElement(elem.ptr, <TacsScalar*> vals.data)
    def setPythonCallback(self, cb):
        self.sptr.setPythonCallback(<PyObject*>cb)
