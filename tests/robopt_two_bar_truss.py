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

from pchaos.parameters import ParameterFactory, ParameterContainer

# Optimization settings
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile'  , type=str, default='detopt.log', help='log file for optimizer')
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
    qmap = pc.getQuadraturePointsWeights({0:10, 1:10, 2:10})
        
    return qmap

# Create the rosenbrock function class
class TwoBarTrussOpt:
    def __init__(self, comm, truss, k):
        self.comm    = comm
        self.problem = truss
        self.nvars   = 3
        self.ncon    = 3
        self.k       = k
        
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

        # Get the quadrature map for the current x
        quadmap = getCollocationPoints(x)
        
        # Find mean
        fmean = 0.0
        fqlist = []
        for q in quadmap.keys():
            yq = quadmap[q]['Y']
            wq = quadmap[q]['W']
            fq = self.problem.getTrussWeight(a1=yq[0], a2=yq[1], h=yq[2])
            fqlist.append(fq)
            fmean += wq*fq

        print fmean
        stop

        fobj = fmean + fvar
        
        # Store the objective function value
        # fobj = self.problem.getTrussWeight(a1=x[0], a2=x[1], h=x[2])
        
        # Store the constraint values
        con = [0.0]*self.ncon
        con[0] = self.problem.getBucklingFirstBar(a1=x[0], a2=x[1], h=x[2])
        con[1] = self.problem.getFailureFirstBar(a1=x[0], a2=x[1], h=x[2])
        con[2] = self.problem.getFailureSecondBar(a1=x[0], a2=x[1], h=x[2])


        
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
        g = [0.0]*self.nvars
        g[0:self.nvars] = self.problem.getTrussWeightDeriv(a1=x[0], a2=x[1], h=x[2])
        
        # Set the constraint gradient
        A = np.zeros([self.ncon, self.nvars])
        A[0,0:self.nvars] = self.problem.getBucklingFirstBarDeriv(a1=x[0], a2=x[1], h=x[2])
        A[1,0:self.nvars] = self.problem.getFailureFirstBarDeriv(a1=x[0], a2=x[1], h=x[2])
        A[2,0:self.nvars] = self.problem.getFailureSecondBarDeriv(a1=x[0], a2=x[1], h=x[2])
        
        return g, A, fail
    
if __name__ == "__main__":

    print "running deterministic optimization "

    # Physical problem
    rho = 0.2836  # lb/in^3
    L   = 5.0     # in
    P   = 25000.0 # lb
    E   = 30.0e6  # psi
    ys  = 36260.0 # psi
    fs  = 1.5
    dtruss = TwoBarTruss(rho, L, P, E, ys, fs)

    # Optimization Problem
    optproblem = TwoBarTrussOpt(MPI.COMM_WORLD, dtruss)
    opt_prob = Optimization(args.logfile, optproblem.evalObjCon)
    
    # Add functions
    opt_prob.addObj('weight')
    opt_prob.addCon('buckling-bar1', type='i')
    opt_prob.addCon('failure-bar1' , type='i')
    opt_prob.addCon('failure-bar2' , type='i')
    
    # Add variables
    opt_prob.addVar('area-1', type='c', value= 1.0, lower= 1.0e-3, upper= 100.0)
    opt_prob.addVar('area-2', type='c', value= 1.0, lower= 1.0e-3, upper= 100.0)
    opt_prob.addVar('height', type='c', value= 1.0, lower= 1.0   , upper= 100.0)
    
    # Optimization algorithm
    if args.algorithm == 'ALGENCAN':
        opt = ALGENCAN()
        opt.setOption('iprint',2)
        opt.setOption('epsfeas',1e-4)
        opt.setOption('epsopt',1e-3)
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
