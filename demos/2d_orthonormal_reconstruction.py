import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from pspace.core import (
    CoordinateFactory, CoordinateSystem,
    BasisFunctionType, PolyFunction
)


# --- Your snippet ---
cf = CoordinateFactory()
cs = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)

coord_x = cf.createUniformCoordinate(cf.newCoordinateID(), "x", dict(a=-1, b=1), max_monomial_dof=3)
coord_y = cf.createUniformCoordinate(cf.newCoordinateID(), "y", dict(a=-1, b=1), max_monomial_dof=3)

cs.addCoordinateAxis(coord_x)
cs.addCoordinateAxis(coord_y)
cs.initialize()

# f(x,y) = 1 + 2x + 3y² + 4xy
terms = [
    (1, Counter({})),
    (2, Counter({0:1})),
    (3, Counter({1:2})),
    (4, Counter({0:1, 1:1}))
]
geometry = PolyFunction(terms)

coeffs = cs.decompose(geometry, sparse=True)
print("2D Geometry Coefficients:")
for k, val in coeffs.items():
    print(f"Basis {k}: {val}")

# --- Add visualization ---
# Build grid
x_vals = np.linspace(-1, 1, 50)
y_vals = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x_vals, y_vals)

# Original function f(x,y)
F_orig = 1 + 2*X + 3*Y**2 + 4*X*Y

# Reconstructed from basis
F_recon = np.zeros_like(F_orig, dtype=float)
for k, psi_k in cs.basis.items():
    # Evaluate basis at each grid point
    val_basis = np.ones_like(F_orig, dtype=float)
    for cid, deg in psi_k.items():
        if cid == 0:  # x-axis
            psi_x = np.array(coord_x.psi_y(X, deg), dtype=float)
            val_basis *= psi_x
        elif cid == 1:  # y-axis
            psi_y = np.array(coord_y.psi_y(Y, deg), dtype=float)
            val_basis *= psi_y
    F_recon += np.array(val_basis, dtype=float) * float(coeffs[k])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
c1 = axes[0].contourf(X, Y, F_orig, 20, cmap='viridis')
axes[0].set_title("Original Geometry f(x,y)")
fig.colorbar(c1, ax=axes[0])

c2 = axes[1].contourf(X, Y, F_recon, 20, cmap='viridis')
axes[1].set_title("Spectral Reconstruction")
fig.colorbar(c2, ax=axes[1])

for ax in axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()
