import numpy as np


## def get_tensor_quadrature_index(k, param_nqpts_map):
##     pidx = {}
##     pids  = param_nqpts_map.keys()
##     nqpts = param_nqpts_map.values()


##     if k < nqpt[0]:
##         return [k,0,0]
##     elif nqpt[0] - k:
##         return [
    
##     for (pid,nqpt) in zip(pids,nqpts):
##         print pid, nqpt
##         if k < nqpt:mod(nqpt-k,k) 
##         pidx[pid] = mod(
        
##     return 

def nqpts(deg):
    """
    Return the number of quadrature points necessary to integrate the
    monomial of degree deg.
    """
    return max(deg/2+1,6)

def tensor_indices(phdmap):
    """
    Get basis functions indices based on tensor product
    """
    pids  = phdmap.keys()    # parameter IDs
    pdegs = phdmap.values()  # parameter degrees

    # exclude deterministic terms
    total_tensor_basis_terms = np.prod(pdegs)
    num_vars  = len(pdegs)

    # Initialize map with empty values corresponding to each key
    idx = {}
    for key in range(total_tensor_basis_terms):
        idx[key] = []

    # Add actual values into the map
    if num_vars == 1:
        ctr = 0
        for k0 in range(pdegs[0]):
            idx[k0].append({pids[0]:k0}) # add one element tuple to map
            ctr += 1
    elif num_vars == 2:
        ctr = 0
        for k0 in range(pdegs[0]):
            for k1 in range(pdegs[1]):
                idx[k0+k1].append({pids[0]:k0,
                                   pids[1]:k1}) # add two element tuple to map
                ctr += 1
    elif num_vars == 3:
        ctr = 0
        for k0 in range(pdegs[0]):
            for k1 in range(pdegs[1]):
                for k2 in range(pdegs[2]):
                    idx[k0+k1+k2].append({pids[0]:k0,
                                          pids[1]:k1,
                                          pids[2]:k2}) # add three element tuple to map
                    ctr += 1
    elif num_vars == 4:
        ctr = 0
        for k0 in range(pdegs[0]):
            for k1 in range(pdegs[1]):
                for k2 in range(pdegs[2]):
                    for k3 in range(pdegs[3]):
                        idx[k0+k1+k2+k3].append({pids[0]:k0,
                                                 pids[1]:k1,
                                                 pids[2]:k2,
                                                 pids[3]:k3}) # add four element tuple to map
                        ctr += 1 
    elif num_vars == 5:
        ctr = 0
        for k0 in range(pdegs[0]):
            for k1 in range(pdegs[1]):
                for k2 in range(pdegs[2]):
                    for k3 in range(pdegs[3]):
                        for k4 in range(pdegs[4]):
                            idx[k0+k1+k2+k3+k4].append({pids[0]:k0,
                                                 pids[1]:k1,
                                                 pids[2]:k2,
                                                 pids[3]:k3,
                                                 pids[4]:k4}) # add five element tuple to map
                            ctr += 1
    else:
        print 'fix implementation for more elements in tuple'
        raise

    # Make a flat list
    flat_list = [item for sublist in idx.values() for item in sublist]

    # Convert to map
    term_polynomial_degree = {}
    for k in range(total_tensor_basis_terms):
        term_polynomial_degree[k] = flat_list[k]
    return term_polynomial_degree


