import numpy as np
from pspace.core import ParameterFactory, ParameterContainer
from pspace.plotter import plot_jacobian

# UQ Parameters
NSAMPLES = 8
NTERMS   = 10

# Problem parameter
FREQ     = 9*np.pi
t        = np.linspace(0,10,100)
ksweight = 500.0

def evalFunctions(x,yqq):
    funcs = []
    funcs.append(np.sin(FREQ*x))
    funcs.append(yqq*np.cos(x))
    return np.array(funcs)

def minKS(ksweight, x, weight):
    si = evalFunctions(x)
    a  = np.sum(np.exp(-ksweight*si))    
    ks = -np.log(a)/ksweight
    return ks

def maxKS(ksweight, x, yqq):
    si = evalFunctions(x, yqq)
    m  = np.max(si)
    a  = np.sum(np.exp(ksweight*(si-m)))
    ks = m + np.log(a)/(ksweight)
    return ks

def maxdispks(t, yqq, ksweight):
    return maxKS(ksweight, t, yqq)

######################################################################
# Setup ParameterContainer to UQ with Sampling and Galerkin Methods
######################################################################

'''Create "Parameter" using "Parameter Factory" object'''
pfactory = ParameterFactory()
alpha = pfactory.createNormalParameter('alpha',
                                       dict(mu=1.0, sigma=0.30),
                                       NTERMS)

'''Add "Parameters" into "ParameterContainer"'''
pc = ParameterContainer()
pc.addParameter(alpha)
pc.initialize()

######################################################################
# Sampling
######################################################################

print("\nevaluate KS function using sampling : nsamples = %d \n" % NSAMPLES)


'''Set number of quadrature points to do deterministic evaluation'''
param_nqpts_map = {0:NSAMPLES}
quadmap = pc.getQuadraturePointsWeights(param_nqpts_map)

''' Solve deterministic ODE at each y '''
flist = []
nsamples = 0
for qindex in quadmap.keys():
    nsamples += 1    
    '''Get weights and nodes for this point in Y/Z-domain'''
    yq = quadmap[qindex]['Y']
    wq = quadmap[qindex]['W']         
    ampval = yq[alpha.getParameterID()]
    fval = maxdispks(t, ampval, ksweight)
    flist.append(fval)
    #print ampval, fval
    
'''Find mean'''
fmean = np.full_like(flist[0], 0)
for i in quadmap.keys():
    wq = quadmap[i]['W']
    fmean   += wq*flist[i]
print("\tSampling E[f_ks^max] = %15.4f" % fmean)

'''Find variance'''
fvar =  np.full_like(flist[0], 0)
for i in quadmap.keys():
    wq = quadmap[i]['W']
    fvar += wq*(fmean-flist[i])**2
print("\tSampling V[f_ks^max] = %15.4f" % fvar)

######################################################################
# Projection
######################################################################

print("\nevaluate KS function using projection : nterms = %d \n" % NTERMS)

N = pc.getNumStochasticBasisTerms()
F = np.zeros((N, 1))
for j in range(N):
    pc.initializeQuadrature({0:N+1})
    '''Project 'f' onto psi(j)'''
    pids = pc.getParameters().keys()
    for q in pc.quadrature_map.keys():
        weight = pc.W(q)*pc.evalOrthoNormalBasis(j,q)
        yq = pc.Y(q)[alpha.getParameterID()]        
        F[j] += maxKS(ksweight, t, yq)*weight

'''Compute moments using the coefficients of projection'''
fgmean = F[0]
fgvar  = 0.0
for i in range(1,N):
    fgvar += F[i]*F[i]

print("\tGalerkin E[f_ks^max] = %15.4f" % fgmean)
print("\tGalerkin V[f_ks^max] = %15.4f" % fgvar)
