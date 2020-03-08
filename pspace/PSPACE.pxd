# Typdefs required for either real or complex mode
include "PspaceTypedefs.pxi"

cdef extern from "AbstractParameter.h":
    cdef cppclass AbstractParameter:
        #void quadrature(int npoints, scalar *z, scalar *y, scalar *w)
        scalar basis(scalar z, int d)

cdef class PyAbstractParameter:
    cdef AbstractParameter *ptr

cdef extern from "ParameterFactory.h":
    cdef cppclass ParameterFactory:
        ParameterFactory()
        AbstractParameter* createNormalParameter(scalar mu, scalar sigma, int dmax)
        AbstractParameter* createUniformParameter(scalar a, scalar b, int dmax)
        AbstractParameter* createExponentialParameter(scalar mu, scalar beta, int dmax)

cdef class PyParameterFactory:
    cdef ParameterFactory *ptr

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

cdef class PyParameterContainer:
    cdef ParameterContainer *ptr
