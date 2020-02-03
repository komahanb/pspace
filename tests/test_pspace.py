import pspace.cwrap as uq
import numpy as np

pfactory = uq.PyParameterFactory()
y1 = pfactory.createNormalParameter(mu=1.0,sigma=0.1,dmax=2)
y2 = pfactory.createUniformParameter(a=1.0,b=0.1,dmax=2)
y3 = pfactory.createExponentialParameter(mu=1.0,beta=0.1,dmax=2)

pc = uq.PyParameterContainer(1)
pc.addParameter(y1)
pc.addParameter(y2)
pc.addParameter(y3)

pc.initialize()

for q in range(pc.getNumQuadraturePoints()):
    zq, yq = pc.quadrature(q)
    for k in range(pc.getNumBasisTerms()):
        print("z=", zq, "q=", q, pc.basis(k, zq))

print(pc.getNumBasisTerms())
