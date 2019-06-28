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
from pspace.core import ParameterFactory, ParameterContainer
from pspace.stochastic_utils import tensor_indices, nqpts, sparse

class  StochasticSpring:
    def __init__(self, dspr, pc):
        self.dspr = dspr
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
        F = np.zeros((self.nsdof))
        
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
        U = np.zeros((self.nsdof))

        # All stochastic parameters are assumed to be of degree 1
        # (constant terms)
        dmapf = Counter()
        for pid in self.pc.parameter_map.keys():
            dmapf[pid] = 1        

        J = np.zeros((self.nsdof,self.nsdof))
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

                    jtmp = np.zeros((self.nddof, self.nddof))
                                    
                    # Quadrature Loop
                    for q in self.pc.quadrature_map.keys():
                        
                        # Quadrature node and weight
                        yq = self.pc.Y(q,'name')
                        wq = self.pc.W(q)

                        # Set the paramter values into the element
                        dspr.setStiffness(yq['K'])
  
                        # Create space for fetching deterministic
                        # jacobian, and state vectors that go as input
                        uq = np.zeros((self.nddof))
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

        print(J)
        print(F)
        return U
                
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
K = pfactory.createNormalParameter('K', dict(mu=np.pi/2., sigma=0.1*(np.pi/2.)), 3)

# Add random parameters into a container and initialize
pc = ParameterContainer()
pc.addParameter(K)
pc.initialize()

# Create deterministic and stochastic spring elements
dspr = Spring(k=np.pi/2.)
sspr = StochasticSpring(dspr, pc)

#######################################################################
# Test the system
#######################################################################

# Assemble stochastic force vector from deterministic vector
f = np.pi
U = sspr.solve(0, f)
# E = dspr.F(uhat)
print("force   :", f)
print("disp    :", U)

E = np.zeros((sspr.nsdof))
for i in range(sspr.nterms):
    # Determine num quadrature point required for k-th
    # projection
    pc.initializeQuadrature(
        pc.getNumQuadraturePointsFromDegree(
            pc.basistermwise_parameter_degrees[i]
            )
        )    
    # Quadrature Loop
    ftmp = 0
    for q in pc.quadrature_map.keys():                        
        # Quadrature node and weight
        yq = pc.Y(q,'name')
        wq = pc.W(q)
    
        # Set the paramter values into the element
        dspr.setStiffness(yq['K'])
      
        # Create space for fetching deterministic
        # jacobian, and state vectors that go as input
        uq = np.zeros((1))# dspr.nddof
        for k in range(sspr.nterms):
            psiky = pc.evalOrthoNormalBasis(k,q)
            uq[:] += U[k*1:(k+1)*1]*psiky

        fq = dspr.F(uq)
        
        # Form u for this quadrature point 'y'    
        # Project the determinic element jacobian onto the
        # stochastic basis and place in the global matrix
        psiziw = wq*pc.evalOrthoNormalBasis(i,q)
        
        # F(q,q(y))*psi(y)*rho(y)
        E[i] += fq*psiziw

fmean = E[0]
fvar  = np.sum(E[1:]**2)
fstd  = np.sqrt(fvar)
print("fmean :", E[0], "fvar :", fvar, "fstd :", fstd) 

print("\nk-derivatives... ")

dRdkq = spr.dRdk(uq)
dFdkq = spr.dFdk(uq)

print("dRdk  :", dRdk)
print("dFdk  :", dFdk)

print("\nq-derivatives...")
dRdq = spr.dRdq(uhat)
dFdq = spr.dFdq(uhat)
print("dRdq  :", dRdq)
print("dFdq  :", dFdq)

print("\nadjoint derivative...")
lam = spr.adjoint_solve(uhat)
print("lambda:", lam)
DFDx = dFdk + lam*dRdk
print("Adjoint DFDx  :", DFDx)
print("Exact   DFDx  :", -0.5*uhat*uhat)

stop

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
