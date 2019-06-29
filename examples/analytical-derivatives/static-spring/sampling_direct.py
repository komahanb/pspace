'''
A test script that computes adjoint gradient based on stochastic
galerkin method for the linear system governed by
              R:= k q = f
where q is the state and f is the force and k is the stiffness.

The function of interest is potential energy evaluated as
              F:= E = 1/2 q*k*q
              
The exact derivative dF/dk = -1/2 q*q

Author: Komahan Boopathy
'''
from __future__ import print_function
import numpy as np
from spring import Spring
from pspace.core import ParameterFactory, ParameterContainer

def get_func_deriv(k):
    """
    Get the energy value and its derivative for given k value
    """
    f = np.pi
    spr = Spring(k)
    u = spr.solve(0, f)
    E = spr.F(u)
    dRdk = spr.dRdk(u)
    dFdk = spr.dFdk(u)
    dRdq = spr.dRdq(u)
    dFdq = spr.dFdq(u)
    lam  = spr.direct_solve(u)
    DFDx = dFdk + lam*dFdq
    return E, DFDx

# Create random parameters
pfactory = ParameterFactory()
K = pfactory.createNormalParameter('K', dict(mu=np.pi/2., sigma=0.1*(np.pi/2.)), 1)

# Add random parameters into a container and initialize
pc = ParameterContainer()
pc.addParameter(K)
pc.initialize()

print('deterministic')
f, dfdx = get_func_deriv(np.pi/2.)
print('fvalue :', f, 'dfdx   :', dfdx)

# Number of collocation points (quadrature points) along each
# parameter dimension
quadmap = pc.getQuadraturePointsWeights({0:5})
ymean = yymean = yprimemean = yyprimemean = 0.0
for q in quadmap.keys():
    yq = quadmap[q]['Y']
    wq = quadmap[q]['W']    
    y, yprime = get_func_deriv(k=yq[0])
    ymean += wq*y
    yymean += wq*y**2
    yprimemean += wq*yprime
    yyprimemean += wq*(2*y*yprime)

# Compute moments and their derivatives
fmean = ymean
fvar = yymean - fmean**2
fstd = np.sqrt(fvar)
fmeanprime = yprimemean
fvarprime = yyprimemean - 2.0*fmean*fmeanprime
fstdprime = fvarprime/(2.0*np.sqrt(fvar))
print('stochastic')
print("fmean :", fmean, "fmeanprime", fmeanprime)
print("fvar  :", fvar , "fvar prime", fvarprime)
print("fstd  :", fstd , "fstd prime", fstdprime)
