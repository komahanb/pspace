import math
import numpy as np
import scipy.special as sp

def tensor_indices(nterms):
    """
    Get basis functions indices based on tensor product
    """
    tot_terms = np.prod(nterms)
    num_vars  = len(nterms)

    #print tot_terms, num_vars

    idx = {}
    for key in range(tot_terms):
        idx[key] = []

    ## for term in nterms:
    ##     for k in range(term):
    ##         print k

    ctr = 0
    for i in range(nterms[0]):
        for j in range(nterms[1]):
            for k in range(nterms[2]):
                ctr += 1
                idx[i+j+k].append((i, j, k))

    ## for key in idx:
    ##     print idx[key], len(idx[key])

    flat_list = [item for sublist in idx.values() for item in sublist]

    return flat_list

def _laguerre(z,d):
    """
    Polynomials such that <f(z), g(z)>_{exp(-z)}^{0,inf} = 0
    """
    if d == 0:
        return 1.0
    elif d == 1:
        return 1.0 - z
    else:
        den = d
        return ((2*d-1-z)*laguerre(z,d-1) - (d-1)*laguerre(z,d-2))/den

def _unit_laguerre(z,d):
    """
    Returns unit laguerre polynomial of degree d evaluated at z
    """
    return _laguerre(z,d)

def laguerre(z, d):
    """
    Standard (non-associated) Laguerre polynomial L_d(z).
    Orthogonal under weight exp(-z) on [0, ∞).
    """
    coeffs = [0]*d + [1]                # degree-d monomial
    L = np.polynomial.laguerre.Laguerre(coeffs)
    return L(z)

def unit_laguerre(z, d):
    """
    Orthonormal Laguerre polynomial ψ_d(z).
    """
    return laguerre(z, d) / np.sqrt(1.0)   # weight already normalized

def _hermite(z, d):
    """
    Use recursion to generate probabilist hermite polynomials

    Hermite polynomials are produced using exp(-z^2)/sqrt(2*pi) as the
    weight on trivial monomials on interval [-inf,inf].

    """
    if d == 0:
        return 1.0
    elif d == 1:
        return z
    else:
        return z*hermite(z,d-1) - (d-1)*hermite(z,d-2)

def _unit_hermite(z,d):
    """
    Returns units hermite polynomial of degree n evaluated at z
    """
    return _hermite(z,d)/np.sqrt(math.factorial(d))

def hermite(z, d):
    """
    Probabilists' Hermite polynomial He_d(z).
    Orthogonal under weight exp(-z^2/2) on (-∞, ∞).
    """
    coeffs = [0]*d + [1]
    H = np.polynomial.hermite_e.HermiteE(coeffs)
    return H(z)

def unit_hermite(z, d):
    """
    Orthonormal Hermite polynomial ψ_d(z).
    """
    return hermite(z, d) / np.sqrt(np.math.factorial(d))

def rlegendre(z, d):
    if d == 0:
        return 1.0
    if d == 1:
        return 2.0*z - 1.0
    return ((2*d - 1) * (2*z - 1) * rlegendre(z, d-1)
            - (d - 1) * rlegendre(z, d-2)) / d

def _legendre(z, d):
    """
    Use recursion to generate Legendre polynomials

    Legendre polynomials are produced using rho(z) = 1.0 as the weight
    on trivial monomials in interval [0,1].
    """
    p = 0.0
    for k in range(d+1):
        dp = sp.comb(d,k)*sp.comb(d+k,k)*(-z)**k
        p = p + dp
    return ((-1)**d)*p

def _unit_legendre(z,d):
    return _legendre(z,d)*np.sqrt(2*d+1)

import numpy as np

#=====================================================================#
# Shifted Legendre Basis on [0,1]
#=====================================================================#

def legendre(z, d):
    """
    Shifted Legendre polynomial of degree d on [0,1].
    Defined as P_d^*(z) = P_d(2z - 1), where P_d is the standard Legendre.
    """
    coeffs = [0] * d + [1]          # coefficient vector for degree d
    P = np.polynomial.legendre.Legendre(coeffs)   # P_d(x) on [-1,1]
    return P(2*z - 1)               # shift domain to [0,1]


def unit_legendre(z, d):
    """
    L^2-orthonormal shifted Legendre polynomial on [0,1]:
      ψ_d(z) = sqrt(2d+1) * P_d^*(z).
    """
    return np.sqrt(2*d + 1) * legendre(z, d)


#=====================================================================#
# Shifted Gauss-Legendre Quadrature on [0,1]
#=====================================================================#

def shifted_legendre_quadrature(npts):
    """
    Gauss-Legendre quadrature nodes/weights shifted from [-1,1] to [0,1].
    Integrates ∫_0^1 f(z) dz exactly for deg ≤ 2npts-1.
    """
    x, w = np.polynomial.legendre.leggauss(npts)  # nodes/weights on [-1,1]
    z = 0.5 * (x + 1)                             # map to [0,1]
    w = 0.5 * w                                   # rescale weights
    return z, w

if __name__ == "__main__":

    """
    Test hermite polynomials
    """
    print("    Test Hermite polynomials   ")
    print (unit_hermite(1.2,0), hermite(1.2,0)/np.sqrt(math.factorial(0)))
    print (unit_hermite(1.2,1), hermite(1.2,1)/np.sqrt(math.factorial(1)))
    print (unit_hermite(1.2,2), hermite(1.2,2)/np.sqrt(math.factorial(2)))
    print (unit_hermite(1.2,3), hermite(1.2,3)/np.sqrt(math.factorial(3)))
    print (unit_hermite(1.2,4), hermite(1.2,4)/np.sqrt(math.factorial(4)))

    """
    Test Legendre polynomials
    """

    print("    Test Legendre polynomials   ")
    print (unit_legendre(1.2,0), legendre(1.2,0), rlegendre(1.2,0))
    print (unit_legendre(1.2,1), legendre(1.2,1), rlegendre(1.2,1))
    print (unit_legendre(1.2,2), legendre(1.2,2), rlegendre(1.2,2))
    print (unit_legendre(1.2,3), legendre(1.2,3), rlegendre(1.2,3))
    print (unit_legendre(1.2,4), legendre(1.2,4), rlegendre(1.2,4))


    """
    Test laguerre polynomials
    """
    print("    Test Laguerre polynomials   ")
    print (unit_laguerre(1.2,0), laguerre(1.2,0))
    print (unit_laguerre(1.2,1), laguerre(1.2,1))
    print (unit_laguerre(1.2,2), laguerre(1.2,2))
    print (unit_laguerre(1.2,3), laguerre(1.2,3))
    print (unit_laguerre(1.2,4), laguerre(1.2,4))
