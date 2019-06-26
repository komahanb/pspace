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
from pspace.core import ParameterFactory, ParameterContainer

# Create random parameters
pfactory = ParameterFactory()
K = pfactory.createNormalParameter('k', dict(mu=5.0, sigma=0.5), 2)

# Add random parameters into a container and initialize
pc = ParameterContainer()
pc.addParameter(K)
pc.initialize()

# Number of collocation points (quadrature points) along each
# parameter dimension
qmap = pc.getQuadraturePointsWeights({0:5})

class Spring:
    def __init__(self, k):
        self.k = k
        return
    def setStiffness(self,ymap):
        self.k = ymap['k']
        return
    def R(self, q, f):
        return self.k*q - f
    def dRdq(self, q):
        return self.k
    def dRdk(self, q):
        return q        
    def solve(self, q, f):
        # Fake the linear solve like newton solve with intial zero
        # guess
        for k in range(1):
            q = q - self.R(q,f)/self.dRdq(q)
        return q    
    def F(self, q):
        return 0.5*self.k*q*q
    def dFdq(self, q):
        return self.k*q
    def dFdk(self, q):
        return 0.5*q*q
    
# Test the system
f = np.pi
spr = Spring(k = np.pi/2.)
u = spr.solve(0, f)
E = spr.F(u)
print("force   :", f)
print("disp    :", u)
print("energy  :", E)

print("\nk-derivatives...")
dRdk = spr.dRdk(u)
dFdk = spr.dFdk(u)
print("dRdk  :", dRdk)
print("dFdk  :", dFdk)

print("\nq-derivatives...")
dRdq = spr.dRdq(u)
dFdq = spr.dFdq(u)
print("dRdq  :", dRdq)
print("dFdq  :", dFdq)



