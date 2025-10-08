import numpy as np
from collections import Counter
from pspace.core import (
    CoordinateFactory, CoordinateSystem,
    BasisFunctionType, PolyFunction
)

"""
| Step | Action                                         | Mathematical Equivalent                          |
| ---- | ---------------------------------------------- | ------------------------------------------------ |
| 1    | Build (T_{ij} = \langle \psi_i, \phi_j\rangle) | Change of basis                                  |
| 2    | Define test polynomial (f(x))                  | Known analytic reference                         |
| 3    | Compute ψ-coefficients (forward mode)          | (a^\psi = \langle f,\psi_i\rangle)               |
| 4    | Adjoint reconstruction of (f)                  | Solve (G_\phi a^\phi = T^\top a^\psi)            |
| 5    | Adjoint reconstruction of (\nabla_x f)         | Same system, new RHS (b' = T^\top a^{\psi,(f')}) |
| 6    | Validate against exact coefficients            | Confirms dual consistency                        |

"""

def build_change_of_basis(cs, coord, max_deg):
    monomials = []
    for d in range(0, max_deg + 1):
        monomials.append((1.0, Counter({coord.id: d})))

    basis_size = len(cs.basis)
    num_monos = len(monomials)
    T = np.zeros((basis_size, num_monos))

    for j, (_, degs) in enumerate(monomials):
        f = PolyFunction([(1.0, degs)], coordinates=cs.coordinates)
        coeffs = cs.decompose(f, sparse=True)
        for k, val in coeffs.items():
            T[k, j] = val
    return T, num_monos, basis_size


def gram_phi(max_deg):
    """Gram matrix for φ-basis (monomials) under the uniform probability measure on [-1,1]."""
    N = max_deg
    G = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if (i + j) % 2 == 0:
                G[i, j] = 1.0 / (i + j + 1)
    return G


def test_adjoint_gradient_reconstruction():
    cf = CoordinateFactory()
    cs = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)

    coord_x = cf.createUniformCoordinate(cf.newCoordinateID(), "x",
                                         dict(a=-1, b=1), max_monomial_dof=4)
    cs.addCoordinateAxis(coord_x)
    cs.initialize()

    max_deg = 4

    # Step 1: Build change of basis (forward transform)
    T, num_monos, basis_size = build_change_of_basis(cs, coord_x, max_deg)
    G_phi = gram_phi(max_deg)
    T_inv = np.linalg.pinv(T)

    # Step 2: Define test function f(x) = 1 + 2x + 3x^2 + 4x^3
    terms = [
        (1.0, Counter({})),
        (2.0, Counter({coord_x.id: 1})),
        (3.0, Counter({coord_x.id: 2})),
        (4.0, Counter({coord_x.id: 3}))
    ]
    f = PolyFunction(terms, coordinates=cs.coordinates)

    # Step 3: Forward decomposition (analysis)
    coeffs_psi = cs.decompose(f, sparse=True)
    a_psi = np.array([coeffs_psi.get(k, 0.0) for k in range(basis_size)])

    # Step 4: Adjoint reconstruction of f (RHS₁)
    b_f = T.T @ a_psi
    a_phi_f = np.linalg.solve(G_phi, b_f)

    # Step 5: Adjoint reconstruction of ∇ₓf (RHS₂)
    # f'(x) = 2 + 6x + 12x^2
    terms_grad = [
        (2.0, Counter({})),
        (6.0, Counter({coord_x.id: 1})),
        (12.0, Counter({coord_x.id: 2}))
    ]
    f_grad = PolyFunction(terms_grad, coordinates=cs.coordinates)

    coeffs_psi_grad = cs.decompose(f_grad, sparse=True)
    a_psi_grad = np.array([coeffs_psi_grad.get(k, 0.0) for k in range(basis_size)])

    b_grad = T.T @ a_psi_grad
    a_phi_grad = np.linalg.solve(G_phi, b_grad)

    # Step 6: Compare against true monomial coefficients
    a_phi_true_f = np.array([1, 2, 3, 4, 0])  # padded to size
    a_phi_true_grad = np.array([2, 6, 12, 0, 0])

    assert np.allclose(a_phi_f, a_phi_true_f, atol=1e-6), \
        f"Adjoint reconstruction of f failed: {a_phi_f}"
    assert np.allclose(a_phi_grad, a_phi_true_grad, atol=1e-6), \
        f"Adjoint reconstruction of grad(f) failed: {a_phi_grad}"

    print("[✓] Adjoint reconstruction of f successful.")
    print("[✓] Adjoint reconstruction of ∇ₓf successful.")

if __name__ == "__main__":
    test_adjoint_gradient_reconstruction()
