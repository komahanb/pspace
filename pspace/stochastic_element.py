class BasisFunction(object):
    def __init__(self, basis_data):
        self.id      = basis_data['basis_id']
        self.name    = basis_data['basis_name']
        self.degrees = basis_data['monomial_degrees'] # Counter({0: 1, 1: 0}),

        # add sanity check for whether basis vector is unitary
        self.num_coords = len(self.degrees)

    def __str__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__) + "\n"

    def getBasisFunctionID(self):
        return self.id

    def getBasisFunctionName(self):
        return self.name

class BasisFunctionFactory:
    def __init__(self):
        self.next_id = 0

    def newBasisFunctionID(self):
        bid = self.next_id
        self.next_id += 1
        return bid

    def __str__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__) + "\n"

    def createBasisFunction(self, basis_id, basis_name, monomial_degrees):
        data                     = {}
        data['basis_id']         = basis_id
        data['basis_name']       = basis_name
        data['monomial_degrees'] = monomial_degrees
        return BasisFunction(data)

    def checkConsistency(self, max_degree=5, npoints=20, tol=1e-12, verbose=True):
        """
        Check orthonormality of unit polynomials under quadrature.

        Coordinates
        ----------
        max_degree : int
            Highest polynomial degree to check.
        npoints : int
            Number of quadrature points to use.
        tol : float
            Numerical tolerance for delta_{mn}.
        verbose : bool
            Print results if True.

        Returns
        -------
        ok : bool
            True if all checks pass within tolerance.
        errors : list
            List of (m,n,value) where error > tol.
        """
        # 1D quadrature from this coordinate
        qmap = self.getQuadraturePointsWeights(npoints)
        z = qmap['zq']
        w = qmap['wq']

        errors = []
        ok = True

        # check inner products
        for m in range(max_degree+1):
            pm = self.evalOrthoNormalBasis(z, m)
            for n in range(max_degree+1):
                pn = self.evalOrthoNormalBasis(z, n)
                ip = np.sum(pm*pn*w)  # quadrature inner product
                target = 1.0 if m == n else 0.0
                if abs(ip - target) > tol:
                    ok = False
                    errors.append((m, n, ip))
                    if verbose:
                        print(f"Fail: <phi_{m}, phi_{n}> = {ip:.6e} (expected {target})")
        if verbose and ok:
            print(f"[{self.__class__.__name__}] consistency check passed "
                  f"for degrees ≤ {max_degree} with {npoints} points.")
        return ok, errors

   def getNumQuadraturePointsFromDegree(self, dmap):
        """
        Supply a map whose keys are coordinateids and values are
        monomial degrees and this function will return a map with
        coordinateids as keys and number of corresponding quadrature
        points along the monomial dimension.
        """
        ## pids = dmap.keys()
        ## coord_nqpts_map = Counter()
        ## for pid in pids:
        ##     coord_nqpts_map[pid] = nqpts(dmap[pid])
        ## return coord_nqpts_map

        pids = dmap.keys()
        coord_nqpts_map = Counter()
        for pid in self.coordinates.keys(): #pids:
            coord_nqpts_map[pid] = self.coordinates[pid].monomial_degree #nqpts(dmap[pid])
        return coord_nqpts_map

    def getCoordinateDegreeForBasisTerm(self, coordid, kthterm):
        """
        What is the polynomial degree of the corresponding k-th or
        Hermite/Legendre basis function? For univariate stochastic
        case d == k, but will change for multivariate case based on
        tensor product or other rules used to construct the
        multivariate basis set.
        """
        return self.basistermwise_coordinate_degrees[kthterm][coordid]

    def getMonimialDegreeCoordinates(self):
        degree_map = {}
        for coordid in self.coordinates.keys():
            degree_map[coordid] = self.coordinates[coordid].monomial_degree
        return degree_map # wrap with Counter()

    def getNumBasisElements(self):
        return self.basis_factory.count + 1

    def addCoordinate(self, coordinate):
        # Add coordinate object to the map of coordinates
        self.coordinates[coordinate.getCoordinateID()] = coordinate

        # Increase the number of stochastic terms (tensor pdt rn)
        # self.num_terms = self.num_terms*new_coordinate.monomial_degree

    def initializeQuadrature(self, coord_nqpts_map):
        self.quadrature_map = self.getQuadraturePointsWeights(coord_nqpts_map)
        return

    def W(self, q):
        wmap = self.quadrature_map[q]['W']
        return wmap

    def psi(self, k, zmap):
        coordids = zmap.keys()
        ans = 1.0
        for coordid in coordids:
            # Deterministic ones return one! maybe we can avoid!
            d   = self.getCoordinateDegreeForBasisTerm(coordid, k)
            val = self.getCoordinate(coordid).evalOrthoNormalBasis(zmap[coordid],d)
            ans = ans*val
        return ans

    def Z(self, q, key='pid'):
        if key == 'pid':
            # use pid as key
            return self.quadrature_map[q]['Z']
        else:
            # use name as key
            qmap = self.quadrature_map[q]['Z']
            nmap = {}
            for pid in qmap.keys():
                nmap[self.getCoordinate(pid).coord_name] = qmap[pid]
            return nmap

    def Y(self, q, key='pid'):
        if key == 'pid':
            # use pid as key
            return self.quadrature_map[q]['Y']
        else:
            # use name as key
            qmap = self.quadrature_map[q]['Y']
            nmap = {}
            for pid in qmap.keys():
                nmap[self.getCoordinate(pid).coord_name] = qmap[pid]
            return nmap

    def evalOrthoNormalBasis(self, k, q):
        return self.psi(k, self.Z(q))

    def getQuadraturePointsWeights(self, coord_nqpts_map):
        """
        Return a map of quadrature point index : quadrature data (Y,Z,W).
        Works for arbitrary number of random variables.
        """
        pids  = list(coord_nqpts_map.keys())
        nqpts = list(coord_nqpts_map.values())

        # fetch 1D quadrature maps for each coordinate
        maps = [self.getCoordinate(pid).getQuadraturePointsWeights(n)
                for pid, n in zip(pids, nqpts)]

        # Cartesian product of index ranges
        qmap = {}
        ctr = 0
        for idx_tuple in product(*[range(n) for n in nqpts]):
            yvec, zvec, w = {}, {}, 1.0
            for pid, i, m in zip(pids, idx_tuple, maps):
                yvec[pid] = m['yq'][i]
                zvec[pid] = m['zq'][i]
                w        *= m['wq'][i]
            qmap[ctr] = {'Y': yvec, 'Z': zvec, 'W': w}
            ctr += 1

        return qmap


    def checkConsistency(self, max_degree=None, tol=1e-12, verbose=True):
        """
        Check orthonormality of the multivariate basis functions
        under the container's quadrature. Also prints a table of
        inner products for debugging.

        Coordinates
        ----------
        max_degree : int or None
            Maximum number of basis terms to check. If None, uses
            all available basis terms.
        tol : float
            Numerical tolerance for delta_{ij}.
        verbose : bool
            Print results if True.

        Returns
        -------
        ok : bool
            True if all checks pass within tolerance.
        errors : list
            List of (i,j,value) where error > tol.
        gram : np.ndarray
            Inner product matrix (approximate identity).
        """
        # Ensure quadrature is initialized
        if not hasattr(self, "quadrature_map"):
            nqpts_map = self.getNumQuadraturePoints()
            self.initializeQuadrature(nqpts_map)

        nbasis = self.getNumBasisElements()
        if max_degree is not None:
            nbasis = min(nbasis, max_degree+1)

        gram = np.zeros((nbasis, nbasis))
        errors = []
        ok = True

        # Build Gram matrix
        for i in range(nbasis):
            for j in range(nbasis):
                s = 0.0
                for q in self.quadrature_map.keys():
                    psi_i = self.evalOrthoNormalBasis(i,q)
                    psi_j = self.evalOrthoNormalBasis(j,q)
                    wq    = self.W(q)
                    s += psi_i * psi_j * wq
                gram[i,j] = s
                target = 1.0 if i == j else 0.0
                if abs(s - target) > tol:
                    ok = False
                    errors.append((i, j, s))

        if verbose:
            print(f"[CoordinateSystem] Gram matrix for {nbasis} basis terms:")
            with np.printoptions(precision=3, suppress=True):
                print(gram)

            if ok:
                print(f"[CoordinateSystem] consistency check passed "
                      f"for {nbasis} basis terms.")
            else:
                print(f"[CoordinateSystem] FAILED: {len(errors)} inconsistencies found.")

        return ok, errors, gram

    def getSymmetricNonZeroIndices(self, dmapf):
        nz = {}
        N = self.getNumBasisElements()
        for i in range(N):
            dmapi = self.basistermwise_coordinate_degrees[i]
            for j in range(i,N):
                dmapj = self.basistermwise_coordinate_degrees[j]
                smap = self.sparse(dmapi, dmapj, dmapf)
                if False not in smap.values():
                    dmap = Counter()
                    dmap.update(dmapi)
                    dmap.update(dmapj)
                    dmap.update(dmapf)
                    nqpts_map = self.getNumQuadraturePointsFromDegree(dmap)
                    nz[(i,j)] = nqpts_map
        return nz

    def getSparseJacobian(self, f, dmapf):
        # rename member functions for local readability
        w    = lambda q    : self.W(q)
        psiz = lambda i, q : self.evalOrthoNormalBasis(i,q)

        nzs = self.getSymmetricNonZeroIndices(dmapf)
        N   = self.getNumBasisElements()
        A   = np.zeros((N, N))
        for index, nqpts in nzs.items():
            self.initializeQuadrature(nqpts)
            pids = self.getCoordinates().keys()
            i    = index[0]
            j    = index[1]
            for q in self.quadrature_map.keys():
                val      = w(q)*psiz(i,q)*psiz(j,q)*f(q)
                A[i, j] += val
                A[j, i] += val
        return A

    def getJacobian(self, f, dmapf):
        # rename member functions for local readability
        w    = lambda q    : self.W(q)
        psiz = lambda i, q : self.evalOrthoNormalBasis(i,q)

        N = self.getNumBasisElements()
        A = np.zeros((N, N))
        for i in range(N):
            dmapi = self.basistermwise_coordinate_degrees[i]
            for j in range(N):
                dmapj = self.basistermwise_coordinate_degrees[j]

                dmap = Counter()
                dmap.update(dmapi)
                dmap.update(dmapj)
                dmap.update(dmapf)

                # add up the degree of both participating functions psizi
                # and psizj to determine the total degree of integrand
                nqpts_map = self.getNumQuadraturePointsFromDegree(dmap)
                self.initializeQuadrature(nqpts_map)

                # Loop quadrature points
                pids = self.getCoordinates().keys()
                for q in self.quadrature_map.keys():
                    A[i,j] += w(q)*psiz(i,q)*psiz(j,q)*f(q)
        return A



def index(ii):
    return ii
    if ii == 0:
        return 0
    if ii == 1:
        return 1
    if ii == 2:
        return 3
    if ii == 3:
        return 2

def projectResidual(self, elem, time, res, X, v, dv, ddv):
    """
    Project the elements deterministic residual onto stochastic
    basis and place in global stochastic residual array
    """

    # size of deterministic element state vector
    ndisps = elem.numDisplacements()
    nnodes = elem.numNodes()
    nddof = ndisps*nnodes
    nsdof = ndisps*self.getNumStochasticBasisTerms()

    for i in range(self.getNumStochasticBasisTerms()):

        # Initialize quadrature with number of gauss points
        # necessary for i-th basis entry
        self.initializeQuadrature(
            self.getNumQuadraturePointsFromDegree(
                self.basistermwise_parameter_degrees[i]
                )
            )

        # Quadrature Loop
        rtmp = np.zeros((nddof))

        for q in self.quadrature_map.keys():

            # Set the parameter values into the element
            elem.setParameters(self.Y(q,'name'))

            # Create space for fetching deterministic residual
            # vector
            resq = np.zeros((nddof))
            uq   = np.zeros((nddof))
            udq  = np.zeros((nddof))
            uddq = np.zeros((nddof))

            # Obtain states at quadrature nodes
            for k in range(self.num_terms):
                psiky = self.evalOrthoNormalBasis(k,q)
                uq[:] += v[k*nddof:(k+1)*nddof]*psiky
                udq[:] += dv[k*nddof:(k+1)*nddof]*psiky
                uddq[:] += ddv[k*nddof:(k+1)*nddof]*psiky

            # Fetch the deterministic element residual
            elem.addResidual(time, resq, X, uq, udq, uddq)

            # Project the determinic element residual onto the
            # stochastic basis and place in global residual array
            psiq   = self.evalOrthoNormalBasis(i,q)
            alphaq = self.W(q)
            rtmp[:] += resq*psiq*alphaq

        # Distribute the residual
        for ii in range(nnodes):
            # Local indices
            listart = index(ii)*ndisps
            liend   = (index(ii)+1)*ndisps
            gistart = index(ii)*nsdof + i*ndisps
            giend   = index(ii)*nsdof + (i+1)*ndisps

            # Place in global residul array node by node
            #print(gistart, giend, listart, liend)
            res[gistart:giend] += rtmp[listart:liend]

    # print("res=", res)
    # plot_vector(res, 'stochatic-element-residual.pdf', normalize=True, precision=1.0e-6)

    return

def projectJacobian(self,
                    elem,
                    time, J, alpha, beta, gamma,
                    X, v, dv, ddv):
    """
    Project the elements deterministic jacobian matrix onto
    stochastic basis and place in global stochastic jacobian matrix
    """
    # All stochastic parameters are assumed to be of degree 1
    # (constant terms)
    dmapf = Counter()
    for pid in self.parameter_map.keys():
        dmapf[pid] = 1

    # size of deterministic element state vector
    ndisps = elem.numDisplacements()
    nnodes = elem.numNodes()
    nddof = ndisps*nnodes
    nsdof = ndisps*self.getNumStochasticBasisTerms()

    for i in range(self.getNumStochasticBasisTerms()):
        imap = self.basistermwise_parameter_degrees[i]

        for j in range(self.getNumStochasticBasisTerms()):
            jmap = self.basistermwise_parameter_degrees[j]

            smap = sparse(imap, jmap, dmapf)

            if False not in smap.values():

                dmap = Counter()
                dmap.update(imap)
                dmap.update(jmap)
                dmap.update(dmapf)
                nqpts_map = self.getNumQuadraturePointsFromDegree(dmap)

                # Initialize quadrature with number of gauss points
                # necessary for i,j-th jacobian entry
                self.initializeQuadrature(nqpts_map)

                jtmp = np.zeros((nddof,nddof))

                # Quadrature Loop
                for q in self.quadrature_map.keys():

                    try:
                        elem.setParameters(self.Y(q,'name'))
                    except:
                        print('exception')
                        raise

                    # Create space for fetching deterministic
                    # jacobian, and state vectors that go as input
                    Aq   = np.zeros((nddof,nddof))
                    uq   = np.zeros((nddof))
                    udq  = np.zeros((nddof))
                    uddq = np.zeros((nddof))
                    for k in range(self.num_terms):
                        psiky = self.evalOrthoNormalBasis(k,q)
                        uq[:] += v[k*nddof:(k+1)*nddof]*psiky
                        udq[:] += dv[k*nddof:(k+1)*nddof]*psiky
                        uddq[:] += ddv[k*nddof:(k+1)*nddof]*psiky

                    # Fetch the deterministic element jacobian matrix
                    elem.addJacobian(time, Aq, alpha, beta, gamma, X, uq, udq, uddq)

                    # Project the determinic element jacobian onto the
                    # stochastic basis and place in the global matrix
                    psiziw = self.W(q)*self.evalOrthoNormalBasis(i,q)
                    psizjw = self.evalOrthoNormalBasis(j,q)
                    jtmp[:,:] += Aq*psiziw*psizjw

                # Distribute blocks (16 times)
                for ii in range(0,nnodes):
                    for jj in range(0,nnodes):

                        # Local indices
                        listart = index(ii)*ndisps
                        liend   = (index(ii)+1)*ndisps
                        ljstart = index(jj)*ndisps
                        ljend   = (index(jj)+1)*ndisps

                        gistart = index(ii)*nsdof + i*ndisps
                        giend   = index(ii)*nsdof + (i+1)*ndisps
                        gjstart = index(jj)*nsdof + j*ndisps
                        gjend   = index(jj)*nsdof + (j+1)*ndisps

                        if i == j:
                            J[gistart:giend, gjstart:gjend] += jtmp[listart:liend, ljstart:ljend]
                        else:
                            J[gistart:giend, gjstart:gjend] += jtmp[listart:liend, ljstart:ljend]
                            #J[gjstart:gjend, gistart:giend] += jtmp[listart:liend, ljstart:ljend]

    #print("J=", J)
    #plot_jacobian(J, 'stochatic-element-block.pdf', normalize=True, precision=1.0e-6)

    return

def projectInitCond(self, elem, v, vd, vdd, xpts):
    """
    Project the elements deterministic initial condition onto
    stochastic basis and place in global stochastic init condition
    array
    """

    # size of deterministic element state vector
    ndisps = elem.numDisplacements()
    nnodes = elem.numNodes()
    nddof = ndisps*nnodes
    nsdof = ndisps*self.getNumStochasticBasisTerms()

    for k in range(self.getNumStochasticBasisTerms()):

        # Initialize quadrature with number of gauss points
        # necessary for k-th basis entry
        self.initializeQuadrature(
            self.getNumQuadraturePointsFromDegree(
                self.basistermwise_parameter_degrees[k]
                )
            )

        # Quadrature Loop
        utmp = np.zeros((nddof))
        udtmp = np.zeros((nddof))
        uddtmp = np.zeros((nddof))
        for q in self.quadrature_map.keys():

            # Set the paramter values into the element
            elem.setParameters(self.Y(q,'name'))

            # Create space for fetching deterministic initial
            # conditions
            uq = np.zeros((nddof))
            udq = np.zeros((nddof))
            uddq = np.zeros((nddof))

            # Fetch the deterministic initial conditions
            elem.getInitConditions(uq, udq, uddq, xpts)

            # Project the determinic initial conditions onto the
            # stochastic basis
            psizkw = self.W(q)*self.evalOrthoNormalBasis(k,q)
            utmp += uq*psizkw
            udtmp += udq*psizkw
            uddtmp += uddq*psizkw

        # Distribute values
        for ii in range(nnodes):
            # Local indices
            listart = index(ii)*ndisps
            liend   = (index(ii)+1)*ndisps
            gistart = index(ii)*nsdof + k*ndisps
            giend   = index(ii)*nsdof + (k+1)*ndisps

            # Place in initial condition array node after node
            v[gistart:giend] += utmp[listart:liend]
            vd[gistart:giend] += udtmp[listart:liend]
            vdd[gistart:giend] += uddtmp[listart:liend]

        ## # Replace numbers less than machine precision with zero to
        ## # avoid numerical issues
        ## if clean is True:
        ##     eps = np.finfo(np.float).eps
        ##     v[np.abs(v) < eps] = 0
        ##     vd[np.abs(vd) < eps] = 0
        ##     vdd[np.abs(vdd) < eps] = 0

    return
