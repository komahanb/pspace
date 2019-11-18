import numpy as np

# Problem parameter
freq = 9*np.pi
amp  = np.pi/2

def evalFunctions(x,yqq):
    funcs = []
    funcs.append(np.sin(freq*x))
    funcs.append(yqq*np.cos(x))
    return np.array(funcs)

def minKS(rho, x, weight):
    si = evalFunctions(x)
    # Compute the sum of the KS terms
    #@m  = np.max(si)
    #m = 0
    a  = np.sum(np.exp(-rho*si))    
    ks = -np.log(a)/rho
    
    # This should match the real function as rho increases
    return ks

def maxKS(rho, x, yqq):
    si = evalFunctions(x, yqq)

    # Compute the sum of the KS terms
    m  = np.max(si)
    #m = 0
    a  = np.sum(np.exp(rho*(si-m)))
    ks = m + np.log(a)/(rho)

    #a  = np.sum(np.exp(weight*rho*(si - m)))
    #ks = m + np.log(a)/(rho*weight)
    
    # This should match the real function as rho increases
    return ks

def maxdispks(t, yqq, rho):
    return maxKS(rho, t, yqq)

t = np.linspace(0,10,100)
rho = 500.0
amp = 1.21
print "max disp", maxdispks(t, amp, rho)


print "\nevaluate function using sampling\n"

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from collections import Counter
from timeit import default_timer as timer

from pspace.core import ParameterFactory, ParameterContainer
from pspace.plotter import plot_jacobian

# Create "Parameter" using "Parameter Factory" object
pfactory = ParameterFactory()
alpha = pfactory.createNormalParameter('alpha', dict(mu=1.0, sigma=0.10), 4)

# Add "Parameter" into "ParameterContainer"
pc = ParameterContainer()
pc.addParameter(alpha)

pc.initialize()

param_nqpts_map = {0:10}
quadmap = pc.getQuadraturePointsWeights(param_nqpts_map)

# Solve deterministic ODE at each y
flist = []
nsamples = 0
for qindex in quadmap.keys():
    nsamples += 1    
    # Get weights and nodes for this point in Y/Z-domain
    yq = quadmap[qindex]['Y']
    wq = quadmap[qindex]['W']         
    ampval = yq[alpha.getParameterID()]
    fval = maxdispks(t, ampval, rho)
    flist.append(fval)
    print ampval, fval
    
print "\n"
# Find mean
fmean = np.full_like(flist[0], 0)
for i in quadmap.keys():
    wq = quadmap[i]['W']
    fmean += wq*flist[i]
print "E[f_ks^max] = ", fmean

# Find variance
fvar =  np.full_like(flist[0], 0)
for i in quadmap.keys():
    wq = quadmap[i]['W']
    fvar += wq*(fmean-flist[i])**2
print "V[f_ks^max] = ", fvar
