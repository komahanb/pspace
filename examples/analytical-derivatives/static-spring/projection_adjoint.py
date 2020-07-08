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

from collections import Counter
from spring import Spring

from core import ParameterFactory, ParameterContainer
from stochastic_utils import tensor_indices, nqpts, sparse

class  StochasticSpring:
    def __init__(self, dspr, pc):
        self.dspr = dspr
        self.dtype = type(self.dspr.k)
        self.pc = pc
        self.nddof = 1
        self.nterms = self.pc.getNumStochasticBasisTerms()
        self.nsdof = self.nddof*self.nterms
        return
        
    def solve(self, u, f):
        """
        Fake the linear solve like newton solve with zero initial
        guess
        """
        # Project f and create F    
        F = np.zeros((self.nsdof), dtype = self.dtype)
        
        for k in range(self.nterms):

            # Determine num quadrature point required for k-th
            # projection
            self.pc.initializeQuadrature(
                self.pc.getNumQuadraturePointsFromDegree(
                    self.pc.basistermwise_parameter_degrees[k]
                    )
                )
            
            # Loop through quadrature points
            fktmp = 0.0
            for q in self.pc.quadrature_map.keys():
                # Quadrature node and weight
                yq = self.pc.Y(q,'name')
                wq = self.pc.W(q)

                # Set the paramter values into the element
                dspr.setStiffness(yq['K'])

                # Evaluate function
                fq = f

                # Project the determinic initial conditions onto the
                # stochastic basis
                psizkw = wq*self.pc.evalOrthoNormalBasis(k,q)
                fktmp += fq*psizkw

            # Set into the force vector
            F[k] = fktmp

        # Initial guess
        U = np.zeros((self.nsdof), dtype = self.dtype)

        # All stochastic parameters are assumed to be of degree 1
        # (constant terms)
        dmapf = Counter()
        for pid in self.pc.parameter_map.keys():
            dmapf[pid] = 1        

        J = np.zeros((self.nsdof,self.nsdof), dtype = self.dtype)
        for i in range(self.nterms):            
            imap = self.pc.basistermwise_parameter_degrees[i]
            
            for j in range(self.nterms):                
                jmap = self.pc.basistermwise_parameter_degrees[j]
                smap = sparse(imap, jmap, dmapf)                
                if False not in smap.values():                    
                    dmap = Counter()
                    dmap.update(imap)
                    dmap.update(jmap)
                    dmap.update(dmapf)                    
                    nqpts_map = self.pc.getNumQuadraturePointsFromDegree(dmap)

                    # Initialize quadrature with number of gauss points
                    # necessary for i,j-th jacobian entry
                    self.pc.initializeQuadrature(nqpts_map)

                    jtmp = np.zeros((self.nddof, self.nddof), dtype = self.dtype)
                                    
                    # Quadrature Loop
                    for q in self.pc.quadrature_map.keys():
                        
                        # Quadrature node and weight
                        yq = self.pc.Y(q,'name')
                        wq = self.pc.W(q)

                        # Set the paramter values into the element
                        dspr.setStiffness(yq['K'])
  
                        # Create space for fetching deterministic
                        # jacobian, and state vectors that go as input
                        uq = np.zeros((self.nddof), dtype = self.dtype)
                        for k in range(self.nterms):
                            psiky = self.pc.evalOrthoNormalBasis(k,q)
                            uq[:] += U[k*self.nddof:(k+1)*self.nddof]*psiky
                            
                        # Fetch the deterministic element jacobian matrix
                        Aq = self.dspr.dRdq(uq)
                                    
                        # Project the determinic element jacobian onto the
                        # stochastic basis and place in the global matrix
                        psiziw = wq*self.pc.evalOrthoNormalBasis(i,q)
                        psizjw = self.pc.evalOrthoNormalBasis(j,q)                        
                        jtmp[:,:] += Aq*psiziw*psizjw

                    # Place the entry into jacobian (transposed)
                    J[i*self.nddof:(i+1)*self.nddof, j*self.nddof:(j+1)*self.nddof] += jtmp[:, :]

        # Solve the linear system 
        U = U - np.linalg.solve(J.T, F)

        return U

    def adjoint_solve(self, U):

        # Project f and create F
        dFdQ = np.zeros((self.nsdof), dtype = self.dtype)
        
        for i in range(self.nterms):

            # Determine num quadrature point required for k-th
            # projection
            self.pc.initializeQuadrature(
                self.pc.getNumQuadraturePointsFromDegree(
                    self.pc.basistermwise_parameter_degrees[i]
                    )
                )
            
            # Loop through quadrature points
            fitmp = 0.0
            for q in self.pc.quadrature_map.keys():
                # Quadrature node and weight
                yq = self.pc.Y(q,'name')
                wq = self.pc.W(q)

                # Set the paramter values into the element
                dspr.setStiffness(yq['K'])

                # Create space for fetching deterministic
                # jacobian, and state vectors that go as input
                uq = np.zeros((self.nddof), dtype = self.dtype)
                for k in range(self.nterms):
                    psiky = self.pc.evalOrthoNormalBasis(k,q)
                    uq[:] += U[k*self.nddof:(k+1)*self.nddof]*psiky
                            
                # Evaluate function
                dfdqq = dspr.dFdq(uq)

                # Project the determinic initial conditions onto the
                # stochastic basis
                psiziw = wq*self.pc.evalOrthoNormalBasis(i,q)
                fitmp += dfdqq*psiziw

            # Set into the force vector
            dFdQ[i] = fitmp

        # All stochastic parameters are assumed to be of degree 1
        # (constant terms)
        dmapf = Counter()
        for pid in self.pc.parameter_map.keys():
            dmapf[pid] = 1        

        J = np.zeros((self.nsdof,self.nsdof), dtype = self.dtype)
        for i in range(self.nterms):            
            imap = self.pc.basistermwise_parameter_degrees[i]
            
            for j in range(self.nterms):                
                jmap = self.pc.basistermwise_parameter_degrees[j]
                smap = sparse(imap, jmap, dmapf)                
                if False not in smap.values():                    
                    dmap = Counter()
                    dmap.update(imap)
                    dmap.update(jmap)
                    dmap.update(dmapf)                    
                    nqpts_map = self.pc.getNumQuadraturePointsFromDegree(dmap)

                    # Initialize quadrature with number of gauss points
                    # necessary for i,j-th jacobian entry
                    self.pc.initializeQuadrature(nqpts_map)

                    jtmp = np.zeros((self.nddof, self.nddof), dtype = self.dtype)
                                    
                    # Quadrature Loop
                    for q in self.pc.quadrature_map.keys():
                        
                        # Quadrature node and weight
                        yq = self.pc.Y(q,'name')
                        wq = self.pc.W(q)

                        # Set the paramter values into the element
                        dspr.setStiffness(yq['K'])
  
                        # Create space for fetching deterministic
                        # jacobian, and state vectors that go as input
                        uq = np.zeros((self.nddof), dtype = self.dtype)
                        for k in range(self.nterms):
                            psiky = self.pc.evalOrthoNormalBasis(k,q)
                            uq[:] += U[k*self.nddof:(k+1)*self.nddof]*psiky
                            
                        # Fetch the deterministic element jacobian matrix
                        Aq = self.dspr.dRdq(uq)
                                    
                        # Project the determinic element jacobian onto the
                        # stochastic basis and place in the global matrix
                        psiziw = wq*self.pc.evalOrthoNormalBasis(i,q)
                        psizjw = self.pc.evalOrthoNormalBasis(j,q)                        
                        jtmp[:,:] += Aq*psiziw*psizjw

                    # Place the entry into jacobian (transposed)
                    J[i*self.nddof:(i+1)*self.nddof, j*self.nddof:(j+1)*self.nddof] = jtmp[:, :]

        return - np.linalg.solve(J.T, dFdQ)

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
h = 1.0e-30j
#K = pfactory.createNormalParameter('K', dict(mu= h + np.pi/2., sigma=0.1*(np.pi/2.)), 7)
#K = pfactory.createExponentialParameter('K', dict(mu= h + np.pi/2., beta=0.1*(np.pi/2.)), 7)
K = pfactory.createUniformParameter('K', dict(a = h + 0.9*np.pi/2., b = h + 1.1*np.pi/2.), 7)

# Add random parameters into a container and initialize
pc = ParameterContainer()
pc.addParameter(K)
pc.initialize()

# Create deterministic and stochastic spring elements
dspr = Spring(k=h + np.pi/2.)
sspr = StochasticSpring(dspr, pc)

#######################################################################
# Test the system
#######################################################################

# Assemble stochastic force vector from deterministic vector
f = np.pi
U = sspr.solve(0, f)
# E = dspr.F(uhat)
# print("force   :", f)
# print("disp    :", U)

dtype = type(h)

# Decompose f(y) and dfdx(y) in stochastic basis
E = np.zeros((sspr.nsdof), dtype = dtype)
dFdX = np.zeros((sspr.nsdof), dtype = dtype)
dRdX = np.zeros((sspr.nsdof), dtype = dtype)
for i in range(sspr.nterms):
    # Determine num quadrature point required for k-th
    # projection
    pc.initializeQuadrature(
        pc.getNumQuadraturePointsFromDegree(
            pc.basistermwise_parameter_degrees[i]
            )
        )
    
    # Quadrature Loop
    for q in pc.quadrature_map.keys():                        
        # Quadrature node and weight
        yq = pc.Y(q,'name')
        wq = pc.W(q)
    
        # Set the paramter values into the element
        dspr.setStiffness(yq['K'])
      
        # Create space for fetching deterministic
        # jacobian, and state vectors that go as input
        uq = np.zeros((1), dtype = dtype) # dspr.nddof
        for k in range(sspr.nterms):
            psiky = pc.evalOrthoNormalBasis(k,q)
            uq[:] += U[k*1:(k+1)*1]*psiky

        fq    = dspr.F(uq)
        dfdxq = dspr.dFdk(uq)
        dRdxq = dspr.dRdk(uq)
        
        # Form u for this quadrature point 'y'    
        # Project the determinic element jacobian onto the
        # stochastic basis and place in the global matrix
        psiziw = wq*pc.evalOrthoNormalBasis(i,q)
        
        # F(q,q(y))*psi(y)*rho(y)
        E[i] += fq*psiziw
        dFdX[i] += dfdxq*psiziw
        dRdX[i] += dRdxq*psiziw

fmean = E[0]
fvar  = np.sum(E[1:]**2)
fstd  = np.sqrt(fvar)
# print("E[f]    :", E[0], "V[f]     :", fvar, "S[f] :", fstd) 

EdRdX = dRdX[0]
VdRdX = np.sum(dRdX[1:]**2)
SdRdX = np.sqrt(VdRdX)
# print("E[dRdx] :", EdRdX, "V[dRdx] :", VdRdX, "S[dRdx] : :", SdRdX)

EdFdX = dFdX[0]
VdFdX = np.sum(dFdX[1:]**2)
SdFdX = np.sqrt(VdFdX)
# print("E[dFdx] :", EdFdX, "V[dFdx] :", VdFdX, "S[dFdx] : :", SdFdX)

lam = sspr.adjoint_solve(U)

# Project f and create F
nsdof = sspr.nsdof
nddof = 1
nterms = sspr.nterms

dFdX = np.zeros((nsdof), dtype = dtype)
e2ffprime = np.zeros((nsdof), dtype = dtype)
for i in range(nterms):

    # Determine num quadrature point required for k-th
    # projection
    pc.initializeQuadrature(
        pc.getNumQuadraturePointsFromDegree(
            pc.basistermwise_parameter_degrees[i]
            )
        )
    
    # Loop through quadrature points
    dfdxi = 0.0
    e2ffprimetmp = 0.0
    for q in pc.quadrature_map.keys():
        # Quadrature node and weight
        yq = pc.Y(q,'name')
        wq = pc.W(q)

        # Set the paramter values into the element
        dspr.setStiffness(yq['K'])

        # Create space for fetching deterministic
        # jacobian, and state vectors that go as input
        uq = np.zeros((nddof), dtype = dtype)
        lamq = np.zeros((nddof), dtype = dtype)
        for k in range(nterms):
            psiky = pc.evalOrthoNormalBasis(k,q)
            uq[:] += U[k*nddof:(k+1)*nddof]*psiky
            lamq[:] += lam[k*nddof:(k+1)*nddof]*psiky
                    
        # Evaluate function
        dfdxq = dspr.getAdjointDeriv(uq, lamq)
        fq    = dspr.F(uq)
        
        # Project the determinic initial conditions onto the
        # stochastic basis
        psiziw = wq*pc.evalOrthoNormalBasis(i,q)
        dfdxi += dfdxq*psiziw
        e2ffprimetmp += 2.0*fq*dfdxq*psiziw

    # Set into the force vector
    dFdX[i] = dfdxi
    e2ffprime[i] = e2ffprimetmp

fmeanprime = dFdX[0]
fvarprime = np.sum(2*E[1:]*dFdX[1:])
fvarprime2 = e2ffprime[0] - 2.0*fmean*fmeanprime
fstdprime = fvarprime/(2.0*np.sqrt(fvar))


print(h)
print('stochastic galerkin adjoint')
print("fmean : %15.14f" % np.real(fmean), "fmeanprime : %15.14f" % np.real(fmeanprime))
print("fvar  : %15.14f" % np.real(fvar) , "fvar prime : %15.14f %15.14f" % (np.real(fvarprime), np.real(fvarprime2)))
print("fstd  : %15.14f" % np.real(fstd) , "fstd prime : %15.14f" % np.real(fstdprime))

print("fmeanprime : %15.14f" % (np.imag(fmean)/1.0e-30), "fmeanprime : %15.14f" % np.real(fmeanprime))
print("fvarprime  : %15.14f" % (np.imag(fvar)/1.0e-30) , "fvar prime : %15.14f %15.14f" % (np.real(fvarprime), np.real(fvarprime2)))
print("fstdprime  : %15.14f" % (np.imag(fstd)/1.0e-30) , "fstd prime : %15.14f" % np.real(fstdprime))
