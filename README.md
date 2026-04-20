# pspace — Polynomial Chaos Expansion Library

Pure-Python library for uncertainty quantification via Polynomial Chaos Expansion (PCE).
It provides a coordinate system of random variables, a family of orthonormal polynomial
bases matched to each distribution, exact Gaussian quadrature, and tools for projecting
functions and operators onto the stochastic basis.

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `sympy`

---

## Core concepts

### Random coordinate axes

Each uncertain parameter is a **coordinate axis** carrying its own distribution, domain,
orthonormal polynomial family, and Gauss quadrature rule.

| Distribution | Polynomial basis | Quadrature |
|---|---|---|
| Normal N(μ, σ) | Probabilists' Hermite | Gauss–Hermite |
| Uniform U(a, b) | Shifted Legendre | Gauss–Legendre |
| Exponential Exp(μ, β) | Laguerre | Gauss–Laguerre |

### Polynomial function representation

`PolyFunction` stores a multivariate polynomial as a list of `(coefficient, Counter{axis: degree})`
pairs. This sparse monomial representation is used throughout for exact sparsity detection and
adaptive quadrature degree selection.

### Multi-index basis sets

`CoordinateSystem` manages the full multi-dimensional parameter space.

| Basis type | Construction rule |
|---|---|
| `TENSOR_DEGREE` | Full tensor product of 1-D bases |
| `TOTAL_DEGREE` | All multi-indices with ∑ degrees ≤ p |

### Adaptive sparsity

Before assembling any inner product, the library checks whether `⟨ψ_i, f, ψ_j⟩` can be
non-zero based on the degree structure of `f`. Only admissible pairs are integrated,
giving significant speedups for high-dimensional or high-degree problems.

---

## Quick start

### 1 — Build a coordinate system

```python
from pspace import CoordinateFactory, CoordinateSystem, BasisFunctionType

cf = CoordinateFactory()
cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)

# p0 ~ Normal(mu=1.0, sigma=0.2), up to degree 3
p0 = cf.createNormalCoordinate(cf.newCoordinateID(), "p0",
                                dict(mu=1.0, sigma=0.2), max_monomial_dof=3)
# p1 ~ Uniform(a=0.8, b=1.2), up to degree 2
p1 = cf.createUniformCoordinate(cf.newCoordinateID(), "p1",
                                 dict(a=0.8, b=1.2), max_monomial_dof=2)

cs.addCoordinateAxis(p0)
cs.addCoordinateAxis(p1)
cs.initialize()

print("Basis size:", cs.getNumBasisFunctions())
```

### 2 — Project a function onto the PCE basis (vector decomposition)

`c_k = ⟨f, ψ_k⟩` for every basis function ψ_k.

```python
from collections import Counter
from pspace import PolyFunction

# f(p0, p1) = 2 + 3*p0 + p0*p1
f = PolyFunction([
    (2.0, Counter()),
    (3.0, Counter({p0.id: 1})),
    (1.0, Counter({p0.id: 1, p1.id: 1})),
])

coeffs = cs.decompose(f, sparse=True)   # {basis_id: coefficient}
```

### 3 — Assemble a stochastic Galerkin matrix (matrix decomposition)

`A_ij = ⟨ψ_i, K, ψ_j⟩` — the projected operator matrix for stiffness K(p).

```python
# K(p) = 1 + 0.1*p0  (parameter-dependent stiffness)
K = PolyFunction([
    (1.0, Counter()),
    (0.1, Counter({p0.id: 1})),
])

A = cs.decompose_matrix(K, sparse=True, symmetric=True)
```

### 4 — Assemble and solve a state equation

```python
from pspace import StateEquation

# RHS: F(p) = 1.0 (constant load)
F = PolyFunction([(1.0, Counter())])

eq = StateEquation("spring", K, F, cs)
eq.assemble(sparse=True)
eq.precondition(method="cholesky")
a = eq.solve()   # PCE coefficients of solution U(p)

# Extract mean and variance
mean_U = a[0]
var_U  = sum(a[1:]**2)
```

### 5 — Verify orthonormality

```python
error = cs.check_orthonormality()   # should be < 1e-12
ok, _, gram = cs.checkConsistency(verbose=True)
```

---

## API reference

### `CoordinateFactory`

| Method | Description |
|---|---|
| `newCoordinateID()` | Auto-increment coordinate ID |
| `createNormalCoordinate(id, name, dict(mu, sigma), max_monomial_dof)` | Normal axis |
| `createUniformCoordinate(id, name, dict(a, b), max_monomial_dof)` | Uniform axis |
| `createExponentialCoordinate(id, name, dict(mu, beta), max_monomial_dof)` | Exponential axis |

### `CoordinateSystem`

| Method | Description |
|---|---|
| `addCoordinateAxis(coord)` | Register a coordinate axis |
| `initialize()` | Build multi-index set |
| `getNumBasisFunctions()` | Number of basis terms |
| `decompose(f, sparse, analytic)` | Vector decomposition `c_k = ⟨f, ψ_k⟩` |
| `decompose_matrix(f, sparse, symmetric)` | Matrix decomposition `A_ij = ⟨ψ_i, f, ψ_j⟩` |
| `inner_product(f, g, f_deg, g_deg)` | Generic inner product `⟨f, g⟩` |
| `inner_product_basis(i, j, f, f_deg)` | Basis inner product `⟨ψ_i, f, ψ_j⟩` |
| `build_quadrature(degrees)` | Tensor-product Gauss rule for given degree map |
| `check_orthonormality()` | Returns `‖G − I‖∞` for the Gram matrix |
| `check_decomposition_numerical_symbolic(f)` | Cross-checks quadrature vs Sympy |

### `StateEquation`

| Method | Description |
|---|---|
| `assemble(analytic, sparse, symmetric)` | Build operator matrix and RHS vector |
| `precondition(method)` | `"cholesky"` or `"spectral"` whitening |
| `solve()` | Solve whitened system; returns PCE coefficient vector |
| `condition_number()` | Condition number of operator matrix |

### `PolyFunction`

Constructed as `PolyFunction([(coeff, Counter({axis_id: degree, ...})), ...])`.
- `__call__(Y)` — evaluate at point `Y = {cid: value}`
- `.degrees` — list of `Counter` objects, one per monomial
- `.max_degrees` — `Counter` of max degree per axis

---

## Repository layout

```
pspace/
  core.py                  — coordinate system, decomposition, state equation
  orthogonal_polynomials.py — Hermite, Legendre, Laguerre families
  stochastic_utils.py      — basis generation, degree arithmetic
  __init__.py              — public API

demos/
  demo_matrix_decomposition.py  — sparse vs full matrix assembly
  demo_state_equation.py        — assemble and solve G·a = b

tests/
  test_vector_decomposition.py  — randomized c_k = ⟨f, ψ_k⟩ tests
  test_matrix_decomposition.py  — randomized A_ij = ⟨ψ_i, f, ψ_j⟩ tests
  test_sparsity_logic.py        — sparsity mask correctness
  test_utils.py                 — shared test helpers
```

---

## Running tests

```bash
pytest tests/ -v
```

All 42 tests should pass across Normal, Uniform, and Exponential distributions,
with both tensor-degree and total-degree bases, and both sparse and full assembly paths.

---

## Author

Komahan Boopathy
