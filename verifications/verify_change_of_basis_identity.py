import numpy as np
from collections import Counter

from pspace.core import (
    CoordinateFactory, CoordinateSystem,
    BasisFunctionType, PolyFunction
)

"""
1. Create a 1D Legendre system up to degree 3
2. Defines monomial functions x^0, x^1, x^2, x^3
3. Uses cs.decompose on each monomitl to fill columns of T
4. Verifies T^{-1}T = I (change of basis consistency)
5. Defined O = I
6. checks commutativity O{phi} = T{-1}O{psi}T{1}

"""
def test_change_of_basis_identity():
    cf = CoordinateFactory()
    cs = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)

    # 1D test (extend later to multi-D)
    coord_x = cf.createUniformCoordinate(cf.newCoordinateID(), "x",
                                         dict(a=-1, b=1),
                                         max_monomial_dof=3)
    cs.addCoordinateAxis(coord_x)
    cs.initialize()

    # Collect monomials up to degree 3
    monomials = []
    for d in range(0, 4):
        monomials.append((1.0, Counter({coord_x.id: d})))

    # Build change-of-basis matrix T
    basis_size = len(cs.basis)
    num_monos = len(monomials)
    T = np.zeros((basis_size, num_monos))

    for j, (_, degs) in enumerate(monomials):
        f = PolyFunction([(1.0, degs)], coordinates=cs.coordinates)
        coeffs = cs.decompose(f, sparse=True)
        for k, val in coeffs.items():
            T[k, j] = val

    # Check invertibility
    T_inv = np.linalg.pinv(T)  # pseudo-inverse for robustness
    I_approx = T_inv @ T
    assert np.allclose(I_approx, np.eye(num_monos), atol=1e-12)

    # Define Identity operator
    O_phi = np.eye(num_monos)
    O_psi = np.eye(basis_size)

    # Verify commutativity: O_phi == T^-1 O_psi T
    lhs = O_phi
    rhs = T_inv @ O_psi @ T
    assert np.allclose(lhs, rhs, atol=1e-12)

    print("[Test] Change-of-basis commutes with Identity operator ✓")

if __name__ == "__main__":
    test_change_of_basis_identity()
