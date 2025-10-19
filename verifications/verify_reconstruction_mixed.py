#!/usr/bin/env python3
import numpy as np
from collections import Counter
from pspace.core import BasisFunctionType, PolyFunction, OrthoPolyFunction
from pspace.numeric import CoordinateFactory, NumericCoordinateSystem

# ----------------------------------------------------------------------
# 1. Coordinate system with heterogeneous distributions
# ----------------------------------------------------------------------
cf = CoordinateFactory()
cs = NumericCoordinateSystem(BasisFunctionType.TENSOR_DEGREE)

cx = cf.createNormalCoordinate(cf.newCoordinateID(), "x",
                               dict(mu=0, sigma=1), max_monomial_dof=3)
cy = cf.createUniformCoordinate(cf.newCoordinateID(), "y",
                                dict(a=-1, b=1), max_monomial_dof=3)
cz = cf.createExponentialCoordinate(cf.newCoordinateID(), "z",
                                    dict(mu=0, beta=1), max_monomial_dof=3)

cs.addCoordinateAxis(cx)
cs.addCoordinateAxis(cy)
cs.addCoordinateAxis(cz)
cs.initialize()

print(f"[Info] Number of basis functions = {cs.getNumBasisFunctions()}")

# ----------------------------------------------------------------------
# 2. Define nonlinear coupled polynomial f(x,y,z)
# ----------------------------------------------------------------------
# f(x,y,z) = 1 + 0.5x + 2y + 0.1z + 0.2xy + 0.05xz + 0.03y²
f = PolyFunction([
    (1.0, Counter()),
    (0.5, Counter({cx.id: 1})),
    (2.0, Counter({cy.id: 1})),
    (0.1, Counter({cz.id: 1})),
    (0.2, Counter({cx.id: 1, cy.id: 1})),
    (0.05, Counter({cx.id: 1, cz.id: 1})),
    (0.03, Counter({cy.id: 2}))
], coordinates=cs.coordinates)

# ----------------------------------------------------------------------
# 3. Decompose to get orthogonal coefficients  (f_psi)
# ----------------------------------------------------------------------
a_psi_dict = cs.decompose(f, sparse=True)
nb = cs.getNumBasisFunctions()
a_psi = [float(a_psi_dict[k]) for k in range(nb)]
f_ortho = OrthoPolyFunction(
    [(a_psi[k], cs.basis[k]) for k in range(nb)],
    cs.coordinates
)

# ----------------------------------------------------------------------
# 4. Reconstruct via operator-aware StateEquation (φ-space)
# ----------------------------------------------------------------------
f_phi = cs.reconstruct(f, sparse=True,
                       precondition=True, method="cholesky", tol=0.0)

# ----------------------------------------------------------------------
# 5. Evaluate f, f_ortho, and f_phi at sample points
# ----------------------------------------------------------------------
np.random.seed(42)
samples = []
for _ in range(5):
    p = {
        cx.id: np.random.normal(0, 1),
        cy.id: np.random.uniform(-1, 1),
        cz.id: np.random.exponential(1)
    }
    samples.append(p)

print("\n[Comparison of f, f_ortho, and f_phi at random points]\n")
for k, p in enumerate(samples):
    fv   = float(f(p))
    fpsi = float(f_ortho(p))
    fphi = float(f_phi(p))
    eψ = abs(fv - fpsi)
    eφ = abs(fv - fphi)
    print(f"Sample {k}: "
          f"x={p[cx.id]:+.3f}, y={p[cy.id]:+.3f}, z={p[cz.id]:+.3f}  "
          f"f={fv:.6f}, f_ortho={fpsi:.6f}, f_phi={fphi:.6f},  "
          f"|errψ={eψ:.2e}, errφ={eφ:.2e}")

# ----------------------------------------------------------------------
# 6. Diagnostic summary
# ----------------------------------------------------------------------
errs_phi = [abs(float(f(p)) - float(f_phi(p))) for p in samples]
errs_psi = [abs(float(f(p)) - float(f_ortho(p))) for p in samples]
print("\n[Summary]")
print(f"Max error (φ): {max(errs_phi):.2e},  RMS (φ): {np.sqrt(np.mean(np.square(errs_phi))):.2e}")
print(f"Max error (ψ): {max(errs_psi):.2e},  RMS (ψ): {np.sqrt(np.mean(np.square(errs_psi))):.2e}")
