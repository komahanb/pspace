from __future__ import print_function
import numpy as np

class TwoBarTruss:
    def __init__(self, rho, L, P, E, ys, fs=1.0):
        self.rho = rho
        self.L   = L
        self.fs  = fs
        self.P   = P
        self.E   = E
        self.ys  = ys
        return
    
    def getTrussWeight(self, a1, a2, h):
        l2 = np.sqrt(self.L*self.L + h*h)
        w = self.rho*a1*self.L + self.rho*a2*l2
        return w

    def getTrussWeightDeriv(self, a1, a2, h):
        L = self.L
        rho = self.rho
        l2 = np.sqrt(L*L + h*h)        
        dwdx = np.zeros(3)
        dwdx[0] = rho*L 
        dwdx[1] = rho*l2
        dwdx[2] = rho*h*a2/l2        
        return dwdx
    
    def getBucklingFirstBar(self, a1, a2, h):        
        return 4.0*self.P*self.fs*(self.L**3)/(np.pi*a1*a1*h*self.E) - 1.0
    
    def getBucklingFirstBarDeriv(self, a1, a2, h):
        dwdx = np.zeros(3)
        dwdx[0] = -8.0*self.P*self.fs*(self.L**3)/(np.pi*a1*a1*a1*h*self.E)
        dwdx[1] = 0.0
        dwdx[2] = -4.0*self.P*self.fs*(self.L**3)/(np.pi*a1*a1*h*h*self.E)
        return dwdx

    def getFailureFirstBar(self, a1, a2, h):
        return self.P*self.fs*self.L/(a1*h*self.ys) - 1.0

    def getFailureFirstBarDeriv(self, a1, a2, h):
        dwdx = np.zeros(3)
        dwdx[0] = -1.0*self.P*self.fs*self.L/(a1*a1*h*self.ys)
        dwdx[1] = 0.0
        dwdx[2] = -1.0*self.P*self.fs*self.L/(a1*h*h*self.ys)
        return dwdx

    def getFailureSecondBar(self, a1, a2, h):
        l2 = np.sqrt(self.L*self.L + h*h)
        return self.P*self.fs*l2/(a2*h*self.ys) - 1.0
    
    def getFailureSecondBarDeriv(self, a1, a2, h):
        l2 = np.sqrt(self.L*self.L + h*h)
        L = self.L
        dwdx = np.zeros(3)
        dwdx[0] = 0.0
        dwdx[1] = -1.0*self.P*self.fs*l2/(a2*a2*h*self.ys)
        dwdx[2] = -1.0*L*L*self.P*self.fs/(a2*h*h*self.ys*l2)
        return dwdx
    
if __name__ == "__main__":
    rho = 0.2836  # lb/in^3
    L   = 5.0     # in
    P   = 25000.0 # lb
    E   = 30.0e6  # psi
    ys  = 36260.0 # psi
    fs  = 1.0
    dtruss = TwoBarTruss(rho, L, P, E, ys, fs)

    # Design variable values
    x = np.random.rand(3)    
    a1 = x[0]
    a2 = x[1]
    h  = x[2]

    # Complex perturbation for derivatives
    dx = 1j*1.0e-30
    
    print("function values\n")
    print("weight         : ", dtruss.getTrussWeight(a1, a2, h))
    print("buckling bar 1 : ", dtruss.getBucklingFirstBar(a1, a2, h))
    print("failure  bar 1 : ", dtruss.getFailureFirstBar(a1, a2, h))
    print("failure  bar 2 : ", dtruss.getFailureSecondBar(a1, a2, h))

    print("\ngradient values\n")
    print("weight         : ",
          np.array([dtruss.getTrussWeight(a1 + dx, a2, h).imag/1e-30,
                    dtruss.getTrussWeight(a1, a2 + dx, h).imag/1e-30,
                    dtruss.getTrussWeight(a1, a2, h + dx).imag/1e-30])
          -
          dtruss.getTrussWeightDeriv(a1, a2, h))
    
    print("buckling bar 1 : ",
          np.array([dtruss.getBucklingFirstBar(a1 + dx, a2, h).imag/1e-30,
                    dtruss.getBucklingFirstBar(a1, a2 + dx, h).imag/1e-30,
                    dtruss.getBucklingFirstBar(a1, a2, h + dx).imag/1e-30])
          -
          dtruss.getBucklingFirstBarDeriv(a1, a2, h))

    print("failure  bar 1 : ",
          np.array([dtruss.getFailureFirstBar(a1 + dx, a2, h).imag/1e-30,
                    dtruss.getFailureFirstBar(a1, a2 + dx, h).imag/1e-30,
                    dtruss.getFailureFirstBar(a1, a2, h + dx).imag/1e-30])
          -
          dtruss.getFailureFirstBarDeriv(a1, a2, h)
          )
    
    print("failure  bar 2 : ",
          np.array([dtruss.getFailureSecondBar(a1 + dx, a2, h).imag/1e-30,
                    dtruss.getFailureSecondBar(a1, a2 + dx, h).imag/1e-30,
                    dtruss.getFailureSecondBar(a1, a2, h + dx).imag/1e-30])
          -
          dtruss.getFailureSecondBarDeriv(a1, a2, h))
