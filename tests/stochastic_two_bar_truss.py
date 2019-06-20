from __future__ import print_function
import numpy as np
from two_bar_truss import TwoBarTruss
from pchaos.parameters import ParameterFactory, ParameterContainer

class StochasticTwoBarTruss:
    def __init__(self, dtruss):
        self.dtruss = dtruss
        return
    
    def getCollocationPoints(self, ymean):    
        # Create random parameters
        pfactory = ParameterFactory()
        a1 = pfactory.createNormalParameter('a1', dict(mu=ymean[0], sigma=0.15), 1)
        a2 = pfactory.createNormalParameter('a2', dict(mu=ymean[1], sigma=0.15), 1)
        h  = pfactory.createNormalParameter('h' , dict(mu=ymean[2], sigma=0.10), 1)
    
        # Add random parameters into a container and initialize
        pc = ParameterContainer()
        pc.addParameter(a1)
        pc.addParameter(a2)
        pc.addParameter(h)
        pc.initialize()
    
        # Number of collocation points (quadrature points) along each
        # parameter dimension
        qmap = pc.getQuadraturePointsWeights({0:5, 1:5, 2:5})
    
        return qmap

    def getTrussWeightMoments(self, a1, a2, h):           
        # Find mean and variance of truss weight
        quadmap = self.getCollocationPoints([a1,a2,h])
        expf = 0.0; expff = 0.0;
        for q in quadmap.keys():
            yq = quadmap[q]['Y']
            wq = quadmap[q]['W']
            fq = self.dtruss.getTrussWeight(a1=yq[0], a2=yq[1], h=yq[2])
            expf += wq*fq
            expff += wq*fq**2
        varf = expff - expf**2        
        return expf, varf, np.sqrt(varf)

    def getBucklingFirstBarMoments(self, a1, a2, h):
        # Find mean and variance of truss weight
        quadmap = self.getCollocationPoints([a1,a2,h])
        expf = 0.0; expff = 0.0;
        for q in quadmap.keys():
            yq = quadmap[q]['Y']
            wq = quadmap[q]['W']
            fq = self.dtruss.getBucklingFirstBar(a1=yq[0], a2=yq[1], h=yq[2])
            expf += wq*fq
            expff += wq*fq**2
        varf = expff - expf**2
        return expf, varf, np.sqrt(varf)
    
    def getFailureFirstBarMoments(self, a1, a2, h):
        # Find mean and variance of truss weight
        quadmap = self.getCollocationPoints([a1,a2,h])
        expf = 0.0; expff = 0.0;
        for q in quadmap.keys():
            yq = quadmap[q]['Y']
            wq = quadmap[q]['W']
            fq = self.dtruss.getFailureFirstBar(a1=yq[0], a2=yq[1], h=yq[2])
            expf += wq*fq
            expff += wq*fq**2
        varf = expff - expf**2
        return expf, varf, np.sqrt(varf)

    def getFailureSecondBarMoments(self, a1, a2, h):
        # Find mean and variance of truss weight
        quadmap = self.getCollocationPoints([a1,a2,h])
        expf = 0.0; expff = 0.0;
        for q in quadmap.keys():
            yq = quadmap[q]['Y']
            wq = quadmap[q]['W']
            fq = self.dtruss.getFailureSecondBar(a1=yq[0], a2=yq[1], h=yq[2])
            expf += wq*fq
            expff += wq*fq**2
        varf = expff - expf**2
        return expf, varf, np.sqrt(varf)
    
    def getTrussWeightMomentsDeriv(self, a1, a2, h):
        """
        #-------------------------------------------------------------#
        # derivative of mean
        #-------------------------------------------------------------#
        # d/dx(E[f(x)]) = E[d/dx(f(x))]

        #-------------------------------------------------------------#
        # variance and its derivative
        #-------------------------------------------------------------#
        # V[f(x)]       = E[f(x)*f(x)] - E[f(x)]*E[f(x)]
        # d/dx(V[f(x)]) = E[2*f(x)*d/dx(f(x))] - 2*E[f(x)]*d/dx(E[f(x)])
        #               = E[2*f(x)*d/dx(f(x))] - 2*E[f(x)]*E[d/dx(f(x))]
        #-------------------------------------------------------------#        
        """
        
        quadmap = self.getCollocationPoints([a1,a2,h])        
        ymean = yymean = yprimemean = yyprimemean = 0.0
        for q in quadmap.keys():
            yq = quadmap[q]['Y']
            wq = quadmap[q]['W']
            
            y = self.dtruss.getTrussWeight(a1=yq[0], a2=yq[1], h=yq[2])            
            ymean += wq*y
            yymean += wq*y**2
            
            yprime = self.dtruss.getTrussWeightDeriv(a1=yq[0], a2=yq[1], h=yq[2])            
            yprimemean += wq*yprime
            yyprimemean += wq*(2*y*yprime)

        # Compute moments and their derivatives
        fmean = ymean
        fvar = yymean - fmean**2
        fmeanprime = yprimemean
        fvarprime = yyprimemean - 2.0*fmean*fmeanprime
        
        return fmeanprime, fvarprime, fvarprime/(2.0*np.sqrt(fvar))
        
    def getBucklingFirstBarMomentsDeriv(self, a1, a2, h):
        quadmap = self.getCollocationPoints([a1,a2,h])        
        ymean = yymean = yprimemean = yyprimemean = 0.0
        for q in quadmap.keys():
            yq = quadmap[q]['Y']
            wq = quadmap[q]['W']
            
            y = self.dtruss.getBucklingFirstBar(a1=yq[0], a2=yq[1], h=yq[2])            
            ymean += wq*y
            yymean += wq*y**2
            
            yprime = self.dtruss.getBucklingFirstBarDeriv(a1=yq[0], a2=yq[1], h=yq[2])            
            yprimemean += wq*yprime
            yyprimemean += wq*(2*y*yprime)

        # Compute moments and their derivatives
        fmean = ymean
        fvar = yymean - fmean**2
        fmeanprime = yprimemean
        fvarprime = yyprimemean - 2.0*fmean*fmeanprime
        
        return fmeanprime, fvarprime, fvarprime/(2.0*np.sqrt(fvar))
            
    def getFailureFirstBarMomentsDeriv(self, a1, a2, h):
        quadmap = self.getCollocationPoints([a1,a2,h])        
        ymean = yymean = yprimemean = yyprimemean = 0.0
        for q in quadmap.keys():
            yq = quadmap[q]['Y']
            wq = quadmap[q]['W']
            
            y = self.dtruss.getFailureFirstBar(a1=yq[0], a2=yq[1], h=yq[2])            
            ymean += wq*y
            yymean += wq*y**2
            
            yprime = self.dtruss.getFailureFirstBarDeriv(a1=yq[0], a2=yq[1], h=yq[2])            
            yprimemean += wq*yprime
            yyprimemean += wq*(2*y*yprime)

        # Compute moments and their derivatives
        fmean = ymean
        fvar = yymean - fmean**2
        fmeanprime = yprimemean
        fvarprime = yyprimemean - 2.0*fmean*fmeanprime
        
        return fmeanprime, fvarprime, fvarprime/(2.0*np.sqrt(fvar))
            
    def getFailureSecondBarMomentsDeriv(self, a1, a2, h):
        quadmap = self.getCollocationPoints([a1,a2,h])        
        ymean = yymean = yprimemean = yyprimemean = 0.0
        for q in quadmap.keys():
            yq = quadmap[q]['Y']
            wq = quadmap[q]['W']
            
            y = self.dtruss.getFailureSecondBar(a1=yq[0], a2=yq[1], h=yq[2])            
            ymean += wq*y
            yymean += wq*y**2
            
            yprime = self.dtruss.getFailureSecondBarDeriv(a1=yq[0], a2=yq[1], h=yq[2])            
            yprimemean += wq*yprime
            yyprimemean += wq*(2*y*yprime)

        # Compute moments and their derivatives
        fmean = ymean
        fvar = yymean - fmean**2
        fmeanprime = yprimemean
        fvarprime = yyprimemean - 2.0*fmean*fmeanprime
        
        return fmeanprime, fvarprime, fvarprime/(2.0*np.sqrt(fvar))
        
if __name__ == "__main__":
    rho = 0.2836  # lb/in^3
    L   = 5.0     # in
    P   = 25000.0 # lb
    E   = 30.0e6  # psi
    ys  = 36260.0 # psi
    fs  = 1.0
    dtruss = TwoBarTruss(rho, L, P, E, ys, fs)
    struss = StochasticTwoBarTruss(dtruss)

    # Design variable values
    x = np.random.rand(3)    
    a1 = 1.0 + x[0]
    a2 = 2.0 + x[1]
    h  = 5.0 + x[2]

    # Complex perturbation for derivatives
    dx = 1j*1.0e-30
    
    print("function values\n")
    print("weight         : ", struss.getTrussWeightMoments(a1, a2, h))
    print("buckling bar 1 : ", struss.getBucklingFirstBarMoments(a1, a2, h))
    print("failure  bar 1 : ", struss.getFailureFirstBarMoments(a1, a2, h))
    print("failure  bar 2 : ", struss.getFailureSecondBarMoments(a1, a2, h))

    print("\ngradient values\n")
    print("weight         : \n",
          np.array([struss.getTrussWeightMoments(a1 + dx, a2, h),
                    struss.getTrussWeightMoments(a1, a2 + dx, h),
                    struss.getTrussWeightMoments(a1, a2, h + dx)])
          .T.imag/1.0e-30
          -
          np.array([struss.getTrussWeightMomentsDeriv(a1, a2, h)]))

    print("buckling bar 1 : \n",
          np.array([struss.getBucklingFirstBarMoments(a1 + dx, a2, h),
                    struss.getBucklingFirstBarMoments(a1, a2 + dx, h),
                    struss.getBucklingFirstBarMoments(a1, a2, h + dx)])
          .T.imag/1.0e-30
          -
          np.array([struss.getBucklingFirstBarMomentsDeriv(a1, a2, h)]))

    print("failure bar 1 : \n",
          np.array([struss.getFailureFirstBarMoments(a1 + dx, a2, h),
                    struss.getFailureFirstBarMoments(a1, a2 + dx, h),
                    struss.getFailureFirstBarMoments(a1, a2, h + dx)])
          .T.imag/1.0e-30
          -
          np.array([struss.getFailureFirstBarMomentsDeriv(a1, a2, h)]))

    print("failure bar 2: \n",
          np.array([struss.getFailureSecondBarMoments(a1 + dx, a2, h),
                    struss.getFailureSecondBarMoments(a1, a2 + dx, h),
                    struss.getFailureSecondBarMoments(a1, a2, h + dx)])
          .T.imag/1.0e-30
          -
          np.array([struss.getFailureSecondBarMomentsDeriv(a1, a2, h)]))
