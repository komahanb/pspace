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
from pspace.numeric import ParameterFactory, ParameterContainer

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
    lam = spr.adjoint_solve(u)
    DFDx = dFdk + lam*dRdk
    return E, DFDx

# Create random parameters
pfactory = ParameterFactory()
#K = pfactory.createExponentialParameter('K', dict(mu = np.pi/2., beta=0.1*(np.pi/2.)), 1)
K = pfactory.createNormalParameter('K', dict(mu=np.pi/2., sigma=0.1*(np.pi/2.)), 1)
#K = pfactory.createUniformParameter('K', dict(a=np.pi/2.-0.1, b=np.pi/2.+0.1), 1)

# Add random parameters into a container and initialize
pc = ParameterContainer()
pc.addParameter(K)
pc.initialize()

print('deterministic')
f, dfdx = get_func_deriv(np.pi/2.)
print('fvalue :', f, 'dfdx   :', dfdx)

# Number of collocation points (quadrature points) along each
# parameter dimension
quadmap = pc.getQuadraturePointsWeights({0:15})
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
print('stochastic sampling adjoint')
print("fmean : %15.14f" % fmean, "fmeanprime : %15.14f" % fmeanprime)
print("fvar  : %15.14f" % fvar , "fvar prime : %15.14f" % fvarprime)
print("fstd  : %15.14f" % fstd , "fstd prime : %15.14f" % fstdprime)



###
h = 1.0e-8
pfactory = ParameterFactory()
#K = pfactory.createExponentialParameter('K', dict(mu = np.pi/2., beta=h + 0.1*(np.pi/2.)), 1)
K = pfactory.createNormalParameter('K', dict(mu= np.pi/2., sigma= h + 0.1*(np.pi/2.)), 1)
#K = pfactory.createUniformParameter('K', dict(a= h+np.pi/2.-0.1, b=h+np.pi/2.+0.1), 1)

# Add random parameters into a container and initialize
pc = ParameterContainer()
pc.addParameter(K)
pc.initialize()

print('deterministic')
g, dgdx = get_func_deriv(h + np.pi/2.)
print('gvalue :', g, 'dgdx   :', dgdx)

# Number of collocation points (quadrature points) along each
# parameter dimension
quadmap = pc.getQuadraturePointsWeights({0:15})
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
gmean = ymean
gvar = yymean - gmean**2
gstd = np.sqrt(gvar)
gmeanprime = yprimemean
gvarprime = yyprimemean - 2.0*gmean*gmeanprime
gstdprime = gvarprime/(2.0*np.sqrt(gvar))
print('stochastic sampling adjoint')
print("gmean : %15.14f" % gmean, "fmeanprime : %15.14f" % gmeanprime)
print("gvar  : %15.14f" % gvar , "fvar prime : %15.14f" % gvarprime)
print("gstd  : %15.14f" % gstd , "fstd prime : %15.14f" % gstdprime)


print("%15.14f" % ((gmean-fmean)/h))
print("%15.14f" % ((gvar-fvar)/h))
print("%15.14f" % ((gstd-fstd)/h))
