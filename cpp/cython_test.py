import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from tacs import uq

# Create "Parameter" using "Parameter Factory" object
pfactory = uq.PyParameterFactory()
p1 = pfactory.createNormalParameter(-4.0, 0.5, 0, None, None)
p2 = pfactory.createUniformParameter(-5.0, 4.0, 0, None, None)
p3 = pfactory.createExponentialParameter(6.0, 1.0, 0, None, None)
p4 = pfactory.createUniformParameter(-5.0, 4.0, 0, None, None)
p5 = pfactory.createExponentialParameter(6.0, 1.0, 0, None, None)

# Add "Parameter" into "ParameterContainer"
pc = uq.PyParameterContainer()
pc.addParameter(p1)
pc.addParameter(p2)
pc.addParameter(p3)
pc.addParameter(p4)
pc.addParameter(p5)

pc.initializeBasis([4,4,4,4,4])
pc.initializeQuadrature([5,5,5,5,5])

nterms = pc.getNumBasisTerms()
nquadpts = pc.getNumQuadraturePoints()

print nterms, nquadpts

for k in range(nterms):
    for q in range(nquadpts):
        wq, zq, yq = pc.quadrature(q)
        pc.basis(k,zq)
