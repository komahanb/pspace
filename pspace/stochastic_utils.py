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

def sparse(dmapi, dmapj, dmapf):
    smap = {}
    for key in dmapi.keys():
        if abs(dmapi[key] - dmapj[key]) <= dmapf[key]:
            smap[key] = True
        else:
            smap[key] = False
    return smap

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

def sum_degrees_union_vector(f_degrees, psi_i):
    """Union over all monomials in f with ψ_i: max degree per axis."""
    degs = Counter()
    for f_deg in f_degrees:
        for a in set(f_deg) | set(psi_i):
            degs[a] = max(degs.get(a, 0),
                          f_deg.get(a, 0) + psi_i.get(a, 0))
    return degs

def sum_degrees_union(f_degrees, psi_i, psi_j):
    """Union over all monomials in f: max degree per axis."""
    degs = Counter()
    for f_deg in f_degrees:
        for a in set(f_deg) | set(psi_i) | set(psi_j):
            degs[a] = max(degs.get(a,0),
                          f_deg.get(a,0) + psi_i.get(a,0) + psi_j.get(a,0))
    return degs


if __name__ == '__main__':

    max_degree_params = {0:2, 1:2}

    out = generate_basis_tensor_degree(max_degree_params)
    print(len(out), out)

    out = generate_basis_total_degree(max_degree_params)
    print(len(out), out)

    basis = {
        0: Counter({'x': 0, 'y': 0}),
        1: Counter({'x': 1, 'y': 0}),
        2: Counter({'x': 0, 'y': 1}),
        3: Counter({'x': 1, 'y': 1})
    }

    deg_f = Counter({'x': 1, 'y': 0})

    mask = sparsity_mask(basis, deg_f)

    print(mask)
