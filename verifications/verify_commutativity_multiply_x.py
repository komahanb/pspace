import numpy as np
from collections import Counter
from pspace.numeric import (
    CoordinateFactory,
    BasisFunctionType,
    PolyFunction,
)
from pspace.verify import NumericCoordinateSystem

"""
In phi-space (monomials): O{phi} is a shift matrix
In psi-space (orthonormal Legendre): O{psi} is tridiagonal
The test verifies O{phi} = T{-1}O{psi}T{1}
"""
def build_change_of_basis(cs, coord, max_deg):
    # Collect monomials up to max_deg
    monomials = []
    for d in range(0, max_deg+1):
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

def operator_phi_multiply_x(num_monos):
    """Operator in monomial basis: shift matrix for x·f"""
    O_phi = np.zeros((num_monos, num_monos))
    for j in range(num_monos-1):
        O_phi[j+1, j] = 1.0
    return O_phi

def operator_psi_multiply_x(basis_size):
    """Operator in Legendre orthonormal basis: tridiagonal recurrence"""
    O_psi = np.zeros((basis_size, basis_size))
    for n in range(basis_size):
        if n+1 < basis_size:
            alpha = (n+1) / np.sqrt((2*n+1)*(2*n+3))
            O_psi[n+1, n] = alpha
        if n-1 >= 0:
            beta = n / np.sqrt((2*n-1)*(2*n+1))
            O_psi[n-1, n] = beta
    return O_psi

def projected_O_phi_multiply_x(max_deg):
    # monomial basis φ_j = x^j, 0..N
    N = max_deg
    G = np.zeros((N+1, N+1))     # Gram matrix in φ
    B = np.zeros((N+1, N+1))     # ⟨x φ_j, φ_i⟩

    # Fill G_ij = ⟨x^j, x^i⟩ and B_ij = ⟨x·x^j, x^i⟩ = ⟨x^{j+1}, x^i⟩
    for i in range(N+1):
        for j in range(N+1):
            # moment(m) = ∫_{-1}^1 x^m dx
            def moment(m):
                return 2.0/(m+1) if (m % 2 == 0) else 0.0
            G[i, j] = moment(i + j)
            if j+1 <= 2*N+1:  # always true here
                B[i, j] = moment(i + j + 1)

    # Projected operator: O_phi = G^{-1} B  (Riesz representation in φ)
    O_phi_proj = np.linalg.solve(G, B)
    return O_phi_proj

def test_commutativity_multiply_x():
    cf = CoordinateFactory()
    cs = NumericCoordinateSystem(BasisFunctionType.TENSOR_DEGREE)

    coord_x = cf.createUniformCoordinate(cf.newCoordinateID(), "x",
                                         dict(a=-1, b=1), max_monomial_dof=4)
    cs.addCoordinateAxis(coord_x)
    cs.initialize()

    max_deg = 4
    T, num_monos, basis_size = build_change_of_basis(cs, coord_x, max_deg)

    # Operators in both bases
    # O_phi = operator_phi_multiply_x(num_monos)
    O_psi = operator_psi_multiply_x(basis_size)
    O_phi = projected_O_phi_multiply_x(max_deg)

    # Transform O_psi into φ-basis
    T_inv = np.linalg.pinv(T)
    O_phi_from_psi = T_inv @ O_psi @ T

    print("O_psi = ", O_psi)
    print("O_phi = ", O_phi)
    print("O_phi_from_psi = ", O_phi_from_psi)

    # Compare
    assert np.allclose(O_phi, O_phi_from_psi, atol=1e-6)

    print("[Test] Multiplication by x commutes across bases ✓")

if __name__ == "__main__":
    test_commutativity_multiply_x()
