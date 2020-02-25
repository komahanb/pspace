# Typdefs required for either real or complex mode
include "PspaceTypedefs.pxi"
include "TacsTypedefs.pxi"

# Include the mpi4py header
cdef extern from "mpi-compat.h":
    pass

cdef extern from "AbstractParameter.h":
    cdef cppclass AbstractParameter:
        #void quadrature(int npoints, scalar *z, scalar *y, scalar *w)
        scalar basis(scalar z, int d)

cdef extern from "ParameterFactory.h":
    cdef cppclass ParameterFactory:
        ParameterFactory()
        AbstractParameter* createNormalParameter(scalar mu, scalar sigma, int dmax)
        AbstractParameter* createUniformParameter(scalar a, scalar b, int dmax)
        AbstractParameter* createExponentialParameter(scalar mu, scalar beta, int dmax)

cdef extern from "ParameterContainer.h":
    cdef cppclass ParameterContainer:
        ParameterContainer(int basis_type, int quadrature_type)
        void addParameter(AbstractParameter *param)

        # Evaluate basis at quadrature points
        scalar quadrature(int q, scalar *zq, scalar *yq)
        scalar basis(int k, scalar *z)

        # Accessors
        int getNumBasisTerms()
        int getNumParameters()
        int getNumQuadraturePoints()
        ## void getBasisParamDeg(int k, int *degs)
        ## void getBasisParamMaxDeg(int *pmax)

        # Initiliazation tasks
        void initialize();
        void initializeBasis(const int *pmax)
        void initializeQuadrature(const int *nqpts)

# Typdefs required for either real or complex mode
# include "TacsTypedefs.pxi"

#from TACS cimport *
from tacs.elements cimport *

cdef extern from "TACSStochasticElement.h":
    cdef cppclass TACSStochasticElement(TACSElement):
        TACSStochasticElement( TACSElement *_delem,
                               ParameterContainer *_pc,
                               void (*_update)(TACSElement*, TacsScalar*, void*) )
        TACSElement* getDeterministicElement()
        void updateElement(TACSElement* elem, TacsScalar* vals)
        void setPythonCallback(PyObject *cbptr)

# A simple test element for TACS
cdef extern from "smd.h":
    cdef cppclass SMD(TACSElement):
        SMD( TacsScalar, TacsScalar, TacsScalar)
        void setMass(TacsScalar)
        void setStiffness(TacsScalar)
        void setDamping(TacsScalar)
    
from tacs.functions cimport *

cdef extern from "TACSStochasticFunction.h":
    cdef cppclass TACSStochasticFunction(TACSFunction):
        TACSStochasticFunction( TACSAssembler *tacs,
                                TACSFunction *dfunc,
                                ParameterContainer *pc,
                                int quantityType,
                                int moment_type )
        TacsScalar getFunctionValue()
    
cdef extern from "TACSKSStochasticFunction.h":
    cdef cppclass TACSKSStochasticFunction(TACSFunction):
        TACSKSStochasticFunction( TACSAssembler *tacs,
                                  TACSFunction *dfunc,
                                  ParameterContainer *pc,
                                  int quantityType,
                                  int moment_type,
                                  int ksweight)
        TacsScalar getFunctionValue()

cdef extern from "TACSMutableElement3D.h":
    cdef cppclass TACSMutableElement3D(TACSElement3D):
        TACSMutableElement3D( TACSElementModel *_model,
                              TACSElementBasis *_basis )
    

