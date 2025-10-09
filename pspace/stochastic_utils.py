import numpy as np

from collections import Counter
from itertools import product
import math

def sum_degrees(*counters):
    """Per-axis sum of Counter degrees."""
    out = Counter()
    for c in counters:
        if c is None:
            continue
        out.update(c)  # Counter adds per-key
    return out

def safe_zero_degrees(coord_ids):
    """Counter with zeros for all coord ids."""
    return Counter({cid: 0 for cid in coord_ids})

def generate_basis_adaptive(f_indices, m):
    """Adaptive basis: closure of f's monomials under m-fold products."""
    if m == 0:
        return {(0,) * len(f_indices[0])}
    combos = product(f_indices, repeat=m)
    return {tuple(sum(idx[i] for idx in combo) for i in range(len(combo[0])))
            for combo in combos}

def minnum_quadrature_points(degree):
    """
    Return the number of quadrature points necessary to integrate the
    monomial of degree deg.
    """
    return math.ceil((degree+1)/2)

def generate_basis_tensor_degree(max_degree_params):
    """
    Construct a tensor-product index map for basis functions.

    Parameters
    ----------
    max_degree_params : dict
        Map {param_id : max_degree}, where max_degree is the highest
        polynomial degree to include for that parameter.

    Returns
    -------
    term_polynomial_degree : dict
        Map {basis_index : Counter({pid: degree, ...})}.
        Each entry gives the parameterwise polynomial degrees
        for that basis term.
    """
    pids  = list(max_degree_params.keys())    # parameter IDs
    pdegs = list(max_degree_params.values())  # max degrees (exclusive upper bound)

    total_tensor_basis_terms = int(np.prod(pdegs))

    basis = {}
    k = 0
    for degrees in product(*[range(d+1) for d in pdegs]):
        basis[k] = Counter({pid: deg for pid, deg in zip(pids, degrees)})
        k += 1

    return basis

def generate_basis_total_degree(max_degree_params):
    """
    Construct a total-degree index map for basis functions.

    Parameters
    ----------
    max_degree_params : dict
        Map {param_id : max_degree}, where max_degree is the highest
        polynomial degree allowed *per dimension*.

    Returns
    -------
    basis : dict
        Map {basis_index : Counter({pid: degree, ...})}.
        Each entry gives the parameterwise polynomial degrees
        for that basis term, with sum(degrees) <= max(total_degrees).
    """
    pids  = list(max_degree_params.keys())
    pdegs = list(max_degree_params.values())

    max_total_degree = max(pdegs)

    basis = {}
    k = 0
    for degrees in product(*[range(d+1) for d in pdegs]):
        if sum(degrees) <= max_total_degree:
            basis[k] = Counter({pid: deg for pid, deg in zip(pids, degrees)})
            k += 1

    return basis

def sum_degrees_union_vector(f_degrees, psi_k: Counter) -> Counter:
    """
    Compute required quadrature degree for product f(y)*ψ_k(y).

    Parameters
    ----------
    f_degrees : Counter or list[Counter]
        Degree structures of f. Each Counter is {axis: degree}.
        Can be a single Counter (max degrees) or list of monomial degrees.
    psi_k : Counter
        Degree structure of basis function ψ_k.

    Returns
    -------
    Counter
        Max degree needed per axis to exactly integrate f * ψ_k.

    Notes
    -----
    - For each axis d, the required degree is:
          deg(d) = max over monomials [ f_deg(d) + psi_k(d) ]
    - Ensures enough quadrature order for vector decomposition.

    Example
    -------
    >>> f_degrees = [Counter({0:2, 1:1})]   # y0^2 * y1
    >>> psi_k = Counter({1:2})              # y1^2
    >>> sum_degrees_union_vector(f_degrees, psi_k)
    Counter({0: 2, 1: 3})   # product ~ y0^2 * y1^3
    """
    degs = Counter()

    # Normalize input
    if isinstance(f_degrees, Counter):
        f_degrees = [f_degrees]

    for f_deg in f_degrees:
        axes = set(f_deg) | set(psi_k)
        for a in axes:
            total = f_deg.get(a, 0) + psi_k.get(a, 0)
            degs[a] = max(degs.get(a, 0), total)

    return degs

def sum_degrees_union_matrix(f_degrees, psi_i: Counter, psi_j: Counter) -> Counter:
    """
    Compute required quadrature degree for product f(y)*ψ_i(y)*ψ_j(y).

    Parameters
    ----------
    f_degrees : Counter or list[Counter]
        Degree structures of f. Each Counter is {axis: degree}.
        Can be a single Counter (max degrees) or list of monomial degrees.
    psi_i, psi_j : Counter
        Degree structures of basis functions ψ_i and ψ_j.

    Returns
    -------
    Counter
        Max degree needed per axis to exactly integrate f * ψ_i * ψ_j.

    Notes
    -----
    - For each axis d, the required degree is:
          deg(d) = max over monomials [ f_deg(d) + psi_i(d) + psi_j(d) ]
    - Ensures enough quadrature order for multivariate products.

    Example
    -------
    >>> f_degrees = [Counter({0:2, 1:1})]   # y0^2 * y1
    >>> psi_i = Counter({0:1})              # y0
    >>> psi_j = Counter({1:2})              # y1^2
    >>> sum_degrees_union(f_degrees, psi_i, psi_j)
    Counter({0: 3, 1: 3})   # product ~ y0^3 * y1^3
    """
    degs = Counter()

    # Normalize input
    if isinstance(f_degrees, Counter):
        f_degrees = [f_degrees]

    for f_deg in f_degrees:
        axes = set(f_deg) | set(psi_i) | set(psi_j)
        for a in axes:
            total = f_deg.get(a, 0) + psi_i.get(a, 0) + psi_j.get(a, 0)
            degs[a] = max(degs.get(a, 0), total)

    return degs


def vector_sparsity_mask(coordinate_system, function, sparse: bool, *, sort_indices: bool = False) -> list[int]:
    """
    Canonical ordering of basis indices to visit for vector decompositions.

    Parameters
    ----------
    coordinate_system : CoordinateSystem interface
        Coordinate system providing basis metadata and sparsity helpers.
    function : PolyFunction
        Function being decomposed (provides term degrees).
    sparse : bool
        Whether to respect sparsity masking or iterate over the full basis.
    """
    if sparse:
        mask = coordinate_system.polynomial_vector_sparsity_mask(function.degrees)
    else:
        mask = coordinate_system.basis.keys()
    indices = [int(idx) for idx in mask]
    if sort_indices:
        indices.sort()
    return indices


def matrix_sparsity_mask(
    coordinate_system,
    function,
    sparse: bool,
    symmetric: bool,
    *,
    sort_pairs: bool = False,
) -> list[tuple[int, int]]:
    """
    Canonical ordering of basis index pairs for matrix decompositions.

    Mirrors the numeric-coordinate-system logic but works with any object
    implementing the CoordinateSystem interface.
    """
    if sparse:
        mask = list(coordinate_system.polynomial_sparsity_mask(function.degrees, symmetric=symmetric))
    else:
        basis_ids = list(coordinate_system.basis.keys())
        if symmetric:
            mask = [(i, j) for ii, i in enumerate(basis_ids) for j in basis_ids[ii:]]
        else:
            mask = [(i, j) for i in basis_ids for j in basis_ids]
    pairs = [(int(i), int(j)) for i, j in mask]
    if sort_pairs:
        pairs.sort()
    return pairs
