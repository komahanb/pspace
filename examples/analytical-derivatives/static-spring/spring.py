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

class Spring:
    def __init__(self, k):
        self.k = k
        self.dtype = type(k)
        return
    def setStiffness(self,k):
        self.k = k
        return
    def R(self, q, f):
        return self.k*q - f
    def dRdq(self, q):
        A = np.zeros((1,1), dtype = self.dtype)
        A[0][0] = self.k
        return A
    def dRdk(self, q):
        return q        
    def solve(self, u, f):
        """
        Fake the linear solve like newton solve with zero initial guess
        """
        for k in range(1):
            A = self.dRdq(u)
            b = np.array([self.R(u,f)])
            u = u - np.linalg.solve(A,b)
        return u    
    def F(self, q):
        return 0.5*self.k*q*q
    def dFdq(self, q):
        return self.k*q
    def dFdk(self, q):
        return 0.5*q*q
    def adjoint_solve(self, q):
        A = self.dRdq(q)
        b = np.array([self.dFdq(q)])
        return - np.linalg.solve(A,b)
    def direct_solve(self, q):
        A = self.dRdq(q)
        b = np.array([self.dRdk(q)])
        return - np.linalg.solve(A,b)
    def getAdjointDeriv(self, u, lam):
        DFDx = self.dFdk(u) + lam*self.dRdk(u)
        return DFDx
    def getDirectDeriv(self, u, psi):
        DFDx = self.dFdk(u) + psi*self.dFdq(u)
        return DFDx
if __name__ == "__main__":
    # Test the system
    f = np.pi
    h = 1.0e-30j
    spr = Spring(k = h + np.pi/2.)
    u = spr.solve(0, f)
    E = spr.F(u)
    print("force   :", f)
    print("disp    :", u)
    print("energy  :", E)
    
    print("\nk-derivatives... ")
    dRdk = spr.dRdk(u)
    dFdk = spr.dFdk(u)
    print("dRdk  :", dRdk)
    print("dFdk  :", dFdk)
    
    print("\nq-derivatives...")
    dRdq = spr.dRdq(u)
    dFdq = spr.dFdq(u)
    print("dRdq  :", dRdq)
    print("dFdq  :", dFdq)
    
    print("\nadjoint derivative...")
    lam = spr.adjoint_solve(u)
    print("lambda:", lam)
    DFDx = spr.getAdjointDeriv(u, lam)
    print("Adjoint DFDx  :", DFDx)
    print("Exact   DFDx  :", -0.5*u*u)

    print("\ndirect derivative...")
    lam = spr.direct_solve(u)
    print("lambda:", lam)
    DFDx = spr.getDirectDeriv(u, lam)
    print("Direct  DFDx  :", DFDx)
    print("Exact   DFDx  :", -0.5*u*u)
    print("Complex DFDx  :", np.imag(E)/np.imag(h))
