#!/usr/bin/env python3
import numpy as np
from collections import Counter
from pspace.core import (
    CoordinateFactory, CoordinateSystem,
    BasisFunctionType, PolyFunction, OrthoPolyFunction
)

# Coordinate system: 1D Normal (ψ = Hermite), φ = monomials
cf = CoordinateFactory()
cs = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)
cx = cf.createNormalCoordinate(cf.newCoordinateID(), "x",
                               dict(mu=0, sigma=1), max_monomial_dof=3)
cs.addCoordinateAxis(cx)
cs.initialize()

# f(x) in φ (monomials)
f = PolyFunction([
    (1.0, Counter()),
    (0.5, Counter({cx.id: 1})),
    (0.25, Counter({cx.id: 2}))
], coordinates=cs.coordinates)

# Reconstruct in φ via StateEquation (should equal f)
f_phi = cs.reconstruct(f, sparse=True, analytic=False,
                       precondition=True, method="cholesky", tol=0.0)

# Also form f_psi from ψ-projection for cross-check
a_psi_dict = cs.decompose(f, sparse=True, analytic=False)
nb = cs.getNumBasisFunctions()
a_psi = [a_psi_dict[k] for k in range(nb)]
f_psi = OrthoPolyFunction([(a_psi[k], cs.basis[k]) for k in range(nb)],
                          cs.coordinates)

# Compare at points
xs = np.linspace(-2.0, 2.0, 5)
for x in xs:
    p = {cx.id: float(x)}
    fv   = float(f(p))
    fphv = float(f_phi(p))
    fpsv = float(f_psi(p))
    eφ = abs(fv - fphv)
    eψ = abs(fv - fpsv)
    print(f"x={x:+.2f} | f={fv:.6f}  f_phi={fphv:.6f}  f_psi={fpsv:.6f}  "
          f"| errφ={eφ:.2e}  errψ={eψ:.2e}")
