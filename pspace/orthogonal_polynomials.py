import numpy as np
import math
from scipy import misc as sp

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

def laguerre(z,d):
    """
    Polynomials such that <f(z), g(z)>_{exp(-z)}^{0,inf} = 0
    """
    if d == 0:
        # to support vector return we multiply with zero
        return 1.0 - 0*z
    elif d == 1:
        return 1.0 - z
    else:
        den = d
        return ((2*d-1-z)*laguerre(z,d-1) - (d-1)*laguerre(z,d-2))/den

def unit_laguerre(z,d):
    """
    Returns unit laguerre polynomial of degree d evaluated at z
    """
    return laguerre(z,d)

def hermite(z, d):
    """
    Use recursion to generate probabilist hermite polynomials

    Hermite polynomials are produced using exp(-z^2)/sqrt(2*pi) as the
    weight on trivial monomials on interval [-inf,inf].
    
    """
    if d == 0:
        return 1.0 - 0*z
    elif d == 1:
        return z
    else:
        return z*hermite(z,d-1) - (d-1)*hermite(z,d-2)

def unit_hermite(z,d):
    """
    Returns units hermite polynomial of degree n evaluated at z
    """
    return hermite(z,d)/np.sqrt(math.factorial(d))
    
## def rlegendre(z,d):
##     y = 2*z-1 #(z+1)/2.0
##     if d == 0:
##         return 1.0
##     elif d == 1:
##         return y
##     else:
##         return ((2*(d-1)+1)*y*rlegendre(y,d-1)-(d-1)*rlegendre(y,d-2))/(1.0*d)

def legendre(z, d):
    """
    Use recursion to generate Legendre polynomials

    Hermite polynomials are produced using rho(z) = 1.0 as the weight
    on trivial monomials in interval [0,1].    
    """
    p = 0.0
    for k in range(d+1):
        dp = sp.comb(d,k)*sp.comb(d+k,k)*(-z)**k
        p = p + dp
    return ((-1)**d)*p

def unit_legendre(z,d):
    return legendre(z,d)*np.sqrt(2*d+1)

if __name__ == "__main__":
    z = np.random.rand()
    for t in range(10000):
        for d in range(10):
            print(d,hermite(z,d))
            print(d,legendre(z,d))
            print(d,laguerre(z,d))
            
##         
##     """
##     Test hermite polynomials
##     """
##     z = 1.1
##     print "hermite"
##     for i in range(10):
##         print unit_hermite(z,i), hermite(z,i)
## 
##     print "\nlaguerre"
##     for i in range(10):
##         print unit_laguerre(z,i), laguerre(z,i)
## 
##     print "\nlegendre"
##     for i in range(10):
##         print unit_legendre(z,i), legendre(z,i)
## 
