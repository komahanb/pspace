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
