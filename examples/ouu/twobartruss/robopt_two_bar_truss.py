# Import some utilities
import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt
import sys, traceback

# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time
import pdb

# =============================================================================
# Extension modules
# =============================================================================
from pyOpt import Optimization
from pyOpt import SLSQP
from pyOpt import ALGENCAN

from two_bar_truss import TwoBarTruss
from stochastic_two_bar_truss import StochasticTwoBarTruss

# Optimization settings
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile'  , type=str, default='robopt', help='log file for optimizer')
parser.add_argument('--algorithm', type=str, default='SLSQP', help='SLSQP/ALGENCAN')
args = parser.parse_args()

def getCollocationPoints(ymean):
    
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

# Create the rosenbrock function class
class TwoBarTrussOpt:
    def __init__(self, comm, truss, k):
        self.comm    = comm
        self.problem = truss
        self.nvars   = 3
        self.ncon    = 3

        # Control stochastic objective and constraints
        self.k  = k
        self.w1 = 1.0
        self.w2 = 1.0
        
        # The design history file
        self.x_hist = []

        # Control redundant evalution of cost and constraint
        # gradients. Evaluate the gradient only if X has changed from
        # previous X.
        # self.currentX = None

        # Space for current function and gradient values
        self.fvals = np.zeros(self.ncon+1)
        self.dfdxvals = np.zeros((self.ncon+1)*self.nvars)
        
        return
   
    def evalObjCon(self, x):
        '''
        Evaluate the objective and constraint
        '''
        # Checking whether DVs have gone negative
        assert(np.all(x>0.0) == 1)
        
        # Set the fail flag
        fail = 0

        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Store the objective function value
        fexp , fvar , fstd  = self.problem.getTrussWeightMoments(a1=x[0], a2=x[1], h=x[2])
        g1exp, g1bar, g1std = self.problem.getBucklingFirstBarMoments(a1=x[0], a2=x[1], h=x[2])
        g2exp, g2bar, g2std = self.problem.getFailureFirstBarMoments(a1=x[0], a2=x[1], h=x[2])
        g3exp, g3bar, g3std = self.problem.getFailureSecondBarMoments(a1=x[0], a2=x[1], h=x[2])

        # Store function values
        fobj = self.w1*fexp + self.w2*fvar
        print "fmean, fvar", fexp, fvar
        
        # Store the constraint values
        con = [0.0]*self.ncon
        con[0] = g1exp + self.k*g1std
        con[1] = g2exp + self.k*g2std
        con[2] = g3exp + self.k*g3std
        
        # Return the values
        return fobj, con, fail

    def evalObjConGradient(self, x, g, A):
        '''
        Evaluate the objective and constraint gradient
        '''
        assert(np.all(x>0.0)==1)
        
        # Set the fail flag
        fail = 0
                     
        # Set the objective gradient
        dfexp , dfvar , dfstd  = self.problem.getTrussWeightMomentsDeriv(a1=x[0], a2=x[1], h=x[2])
        dg1exp, dg1bar, dg1std = self.problem.getBucklingFirstBarMomentsDeriv(a1=x[0], a2=x[1], h=x[2])
        dg2exp, dg2bar, dg2std = self.problem.getFailureFirstBarMomentsDeriv(a1=x[0], a2=x[1], h=x[2])
        dg3exp, dg3bar, dg3std = self.problem.getFailureSecondBarMomentsDeriv(a1=x[0], a2=x[1], h=x[2])

        g = [0.0]*self.nvars
        g[0:self.nvars] = self.w1*dfexp + self.w2*dfvar

        # Set the constraint gradient
        A = np.zeros([self.ncon, self.nvars])
        A[0,0:self.nvars] = dg1exp + self.k*dg1std
        A[1,0:self.nvars] = dg2exp + self.k*dg2std
        A[2,0:self.nvars] = dg3exp + self.k*dg3std
        
        return g, A, fail
    
if __name__ == "__main__":

    print "running robust optimization "

    # Physical problem
    rho = 0.2836  # lb/in^3
    L   = 5.0     # in
    P   = 25000.0 # lb
    E   = 30.0e6  # psi
    ys  = 36260.0 # psi
    fs  = 1.5
    dtruss = TwoBarTruss(rho, L, P, E, ys, fs)
    struss = StochasticTwoBarTruss(dtruss)

    # Optimization Problem
    optproblem = TwoBarTrussOpt(MPI.COMM_WORLD, struss, k = 4.0)
    opt_prob = Optimization(args.logfile, optproblem.evalObjCon)
    
    # Add functions
    opt_prob.addObj('weight')
    opt_prob.addCon('buckling-bar1', type='i')
    opt_prob.addCon('failure-bar1' , type='i')
    opt_prob.addCon('failure-bar2' , type='i')
    
    # Add variables
    opt_prob.addVar('area-1', type='c', value= 1.0, lower= 1.0e-3, upper= 2.0)
    opt_prob.addVar('area-2', type='c', value= 1.0, lower= 1.0e-3, upper= 2.0)
    opt_prob.addVar('height', type='c', value= 4.0, lower= 4.0   , upper= 10.0)
    
    # Optimization algorithm
    if args.algorithm == 'ALGENCAN':
        opt = ALGENCAN()
        opt.setOption('iprint',2)
        opt.setOption('epsfeas',1e-6)
        opt.setOption('epsopt',1e-6)
    else:
        opt = SLSQP(pll_type='POA')
        opt.setOption('MAXIT',999)
    
    opt(opt_prob,
        sens_type=optproblem.evalObjConGradient,
        disp_opts=True,
        store_hst=True,
        hot_start=False)
    
    if optproblem.comm.Get_rank() ==0:   
        print opt_prob.solution(0)
        opt_prob.write2file(disp_sols=True)
