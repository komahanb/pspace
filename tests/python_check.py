import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import numpy as np

from pspace.core import ParameterFactory, ParameterContainer

# Create "Parameter" using "Parameter Factory" object
pfactory = ParameterFactory()
c = pfactory.createNormalParameter('c', dict(mu=-4.0, sigma=0.50), 5)
k = pfactory.createUniformParameter('k', dict(a=-5.0, b=4.0), 5)
m = pfactory.createExponentialParameter('m', dict(mu=6.0, beta=1.0), 5)
d = pfactory.createUniformParameter('d', dict(a=-5.0, b=4.0), 5)
e = pfactory.createExponentialParameter('e', dict(mu=6.0, beta=1.0), 5)

# Add "Parameter" into "ParameterContainer"
pc = ParameterContainer()
pc.addParameter(c)
pc.addParameter(k)
pc.addParameter(m)
pc.addParameter(d)
pc.addParameter(e)

pc.initialize()
pc.initializeQuadrature({0:5,1:5,2:5,3:5,4:5})

N = pc.getNumStochasticBasisTerms()
print("Number of basis terms: ", N)

for k in range(N):
    pids = pc.getParameters().keys()
    for q in pc.quadrature_map.keys():
        pc.evalOrthoNormalBasis(k,q)
        pc.W(q)
