# Typdefs required for either real or complex mode
include "PspaceTypedefs.pxi"

cdef extern from "NormalParameter.h":
    cdef cppclass NormalParameter:
        NormalParameter(int pid, scalar mu, scalar sigma)
        #void quadrature(int npoints, scalar *z, scalar *y, scalar *w)
        #scalar basis(scalar z, int d)
