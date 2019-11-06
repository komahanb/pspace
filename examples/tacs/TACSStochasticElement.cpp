#include "TACSStochasticElement.h"

namespace{
  /*
    Predetermine whether the jacobian block is nonzero
  */
  bool nonzero(const int nvars, int *dmapi, int *dmapj, int *dmapk){
    int filter[nvars];
    for( int i = 0; i < nvars; i++ ){
      if (abs(dmapi[i] - dmapj[i]) <= dmapk[i]){
        filter[i] = 1;
      } else {
        filter[i] = 0;
      }
    }
    int prod = 1;
    for( int i = 0; i < nvars; i++ ){
      prod *= filter[i];
    }
    return prod;
  }

  /*
    Place entry into the matrix location
  */
  TacsScalar getElement(TacsScalar *A, int size, int row, int col) {
    // printf(" local entry %d = ", size * row + col);
    return A[size * row + col];
  };

  /*
    Add entry into the matrix location
  */
  void addElement(TacsScalar *J, int size, int row, int col, TacsScalar value) {
    //  printf(" global entry  %d \n", size * row + col);
    J[size * row + col] += value;
  };

  /*
    Display the matrix sparsity pattern
  */
  void printSparsity( TacsScalar *J, int size,  double tol = 1.0e-12 ){
    for (int ii = 0; ii < size; ii++){
      printf("%2d ", ii);
      for (int jj = 0; jj < size; jj++){
        if (abs(getElement(J, size, ii, jj)) > tol) {
          printf("%2s", "x");
        } else {
          printf("%2s", " ");
        }
      }
      printf("\n");
    }
  }
}

TACSStochasticElement::TACSStochasticElement( TACSElement *_delem,
                                              ParameterContainer *_pc,
                                              void (*_update)(TACSElement*, TacsScalar*) ){
  // Store the deterministic element
  delem = _delem;
  delem->incref();

  // Set callback for element update
  update = _update;

  // Set the component numner of this element
  setComponentNum(delem->getComponentNum());

  // Parameter container
  pc = _pc;

  // Set number of dofs
  num_nodes     = delem->getNumNodes();
  vars_per_node = pc->getNumBasisTerms()*delem->getVarsPerNode();
}

TACSStochasticElement::~TACSStochasticElement(){
  delem->decref();
}

/*
  TACS Element member functions
*/
int TACSStochasticElement::getVarsPerNode() {
  return vars_per_node;
}

int TACSStochasticElement::getNumNodes() {
  return num_nodes;
}

/*
  Return the Initial conditions after projection
*/
void TACSStochasticElement::getInitConditions( int elemIndex,
                                               const TacsScalar X[],
                                               TacsScalar v[],
                                               TacsScalar dv[],
                                               TacsScalar ddv[] ){
  // Deterministic information
  const int ndnvars = delem->getVarsPerNode();
  const int nddof   = delem->getNumVariables();
  const int nsdof   = this->getNumVariables();
  const int nsterms = pc->getNumBasisTerms();

  // Space for quadrature points and weights
  const int nsparams = pc->getNumParameters();
  double *zq = new double[nsparams];
  double *yq = new double[nsparams];
  double wq;

  // Create space for states
  TacsScalar *uq    = new TacsScalar[nddof];
  TacsScalar *udq   = new TacsScalar[nddof];
  TacsScalar *uddq  = new TacsScalar[nddof];

  TacsScalar *utmpk    = new TacsScalar[nddof];
  TacsScalar *udtmpk   = new TacsScalar[nddof];
  TacsScalar *uddtmpk  = new TacsScalar[nddof];

  const int nqpts = pc->getNumQuadraturePoints();

  //  Projection of initial conditions and return
  for (int k = 0; k < nsterms; k++){

    memset(utmpk  , 0, nddof*sizeof(TacsScalar));
    memset(udtmpk , 0, nddof*sizeof(TacsScalar));
    memset(uddtmpk, 0, nddof*sizeof(TacsScalar));

    for (int q = 0; q < nqpts; q++){
      // Get the quadrature points and weights
      wq = pc->quadrature(q, zq, yq);

      // Set the parameter values into the element
      updateElement(delem, yq);

      // reset the states to zero
      memset(uq  , 0, nddof*sizeof(TacsScalar));
      memset(udq , 0, nddof*sizeof(TacsScalar));
      memset(uddq, 0, nddof*sizeof(TacsScalar));

      // Fetch the deterministic element residual
      delem->getInitConditions(elemIndex, X, uq, udq, uddq);

      //  Project the determinic states onto the stochastic basis and
      //  place in global state array
      double scale = pc->basis(k,zq)*wq;
      for (int c = 0; c < nddof; c++){
        utmpk[c] += uq[c]*scale;
        udtmpk[c] += udq[c]*scale;
        uddtmpk[c] += uddq[c]*scale;
      }
    } // quadrature

    // Store k-th projected ICs into stochastic array
    int ptr = k*nddof;
    for (int ii = 0; ii < nddof; ii++){
      v[ptr+ii] = utmpk[ii];
      dv[ptr+ii] = udtmpk[ii];
      ddv[ptr+ii] = uddtmpk[ii];
    }
  }

  // clear the heap
  delete [] uq;
  delete [] udq;
  delete [] uddq;
  delete [] utmpk;
  delete [] udtmpk;
  delete [] uddtmpk;
  delete [] zq;
  delete [] yq;
}

/*
  Compute the residual of the governing equations
*/
void TACSStochasticElement::addResidual( int elemIndex,
                                         double time,
                                         const TacsScalar X[],
                                         const TacsScalar v[],
                                         const TacsScalar dv[],
                                         const TacsScalar ddv[],
                                         TacsScalar res[] ){
  // Deterministic information
  const int ndnvars = delem->getVarsPerNode();
  const int nddof   = delem->getNumVariables();
  const int nsdof   = this->getNumVariables();
  const int nsterms = pc->getNumBasisTerms();

  // Space for quadrature points and weights
  const int nsparams = pc->getNumParameters();
  double *zq = new double[nsparams];
  double *yq = new double[nsparams];
  double wq;

  // Create space for fetching deterministic residuals and states
  TacsScalar *uq    = new TacsScalar[nddof];
  TacsScalar *udq   = new TacsScalar[nddof];
  TacsScalar *uddq  = new TacsScalar[nddof];
  TacsScalar *resq  = new TacsScalar[nddof];
  TacsScalar *rtmpi = new TacsScalar[nddof];

  const int nqpts = pc->getNumQuadraturePoints();

  for (int i = 0; i < nsterms; i++){

    memset(rtmpi, 0, nddof*sizeof(TacsScalar));

    for (int q = 0; q < nqpts; q++){

      // Get the quadrature points and weights
      wq = pc->quadrature(q, zq, yq);

      // Set the parameter values into the element
      updateElement(delem, yq);

      // reset the states and residuals
      memset(resq, 0, nddof*sizeof(TacsScalar));
      memset(uq  , 0, nddof*sizeof(TacsScalar));
      memset(udq , 0, nddof*sizeof(TacsScalar));
      memset(uddq, 0, nddof*sizeof(TacsScalar));

      // Evaluate the basis at quadrature node and form the state
      // vectors
      for (int k = 0; k < nsterms; k++){
        double psikz = pc->basis(k,zq);
        int ptr = k*nddof;
        for (int c = 0; c < nddof; c++){
          uq[c]   += v[ptr+c]*psikz;
          udq[c]  += dv[ptr+c]*psikz;
          uddq[c] += ddv[ptr+c]*psikz;
        }
      }

      // Fetch the deterministic element residual
      delem->addResidual(elemIndex, time, X, uq, udq, uddq, resq);

      //  Project the determinic element residual onto the
      //  stochastic basis and place in global residual array
      double scale = pc->basis(i,zq)*wq;
      for (int c = 0; c < nddof; c++){
        rtmpi[c] += resq[c]*scale;
      }
    } // quadrature

    // Store i-th projected Residual into stochastic array
    int ptr = i*nddof;
    for (int ii = 0; ii < nddof; ii++){
      res[ptr+ii] += rtmpi[ii];
    }

  } // end nsterms

  // clear the heap
  delete [] rtmpi;
  delete [] resq;
  delete [] uq;
  delete [] udq;
  delete [] uddq;
  delete [] zq;
  delete [] yq;
}

void TACSStochasticElement::addJacobian( int elemIndex,
                                         double time,
                                         TacsScalar alpha,
                                         TacsScalar beta,
                                         TacsScalar gamma,
                                         const TacsScalar X[],
                                         const TacsScalar v[],
                                         const TacsScalar dv[],
                                         const TacsScalar ddv[],
                                         TacsScalar res[],
                                         TacsScalar mat[] ){
  // Call the residual implementation
  addResidual(elemIndex, time, X, v, dv, ddv, res);

  // Deterministic information
  const int ndnvars = delem->getVarsPerNode();
  const int nddof   = delem->getNumVariables();
  const int nsdof   = this->getNumVariables();
  const int nsterms = pc->getNumBasisTerms();
  const int nnodes  = getNumNodes();

  printf("nsdof = %d, nddof = %d, nnodes = %d, ndnvars = %d, nsterms = %d \n",
         nsdof, nddof, nnodes, ndnvars, nsterms);

  // Space for quadrature points and weights
  const int nsparams = pc->getNumParameters();
  double *zq = new double[nsparams];
  double *yq = new double[nsparams];
  double wq;

  // polynomial degrees
  int dmapi[nsparams], dmapj[nsparams], dmapf[nsparams];

  // Assume all parameters to be of degree atmost 3 (faster) // fix
  for (int i = 0; i <nsparams; i++){
    dmapf[i] = 3;
  }

  // Create space for fetching deterministic residuals and states
  TacsScalar *uq    = new TacsScalar[nddof];
  TacsScalar *udq   = new TacsScalar[nddof];
  TacsScalar *uddq  = new TacsScalar[nddof];
  TacsScalar *A     = new TacsScalar[nddof*nddof];
  TacsScalar *resq  = new TacsScalar[nddof];

  const int nqpts = pc->getNumQuadraturePoints();

  for (int i = 0; i < nsterms; i++){

    pc->getBasisParamDeg(i, dmapi);

    for (int j = 0; j < nsterms; j++){

      pc->getBasisParamDeg(j, dmapj);

      if (1){ // nonzero(nsparams, dmapi, dmapj, dmapf)){

        memset(A, 0, nddof*nddof*sizeof(TacsScalar));

        for (int q = 0; q < nqpts; q++){

          // Get quadrature points
          wq = pc->quadrature(q, zq, yq);

          // Set the parameter values into the element
          this->updateElement(this->delem, yq);

          // reset the states
          memset(uq  , 0, nddof*sizeof(TacsScalar));
          memset(udq , 0, nddof*sizeof(TacsScalar));
          memset(uddq, 0, nddof*sizeof(TacsScalar));

          // Evaluate the basis at quadrature node and form the state
          // vectors
          for (int k = 0; k < nsterms; k++){
            double psikz = pc->basis(k,zq);
            int ptr = k*nddof;
            for (int c = 0; c < nddof; c++){
              uq[c]   += v[ptr+c]*psikz;
              udq[c]  += dv[ptr+c]*psikz;
              uddq[c] += ddv[ptr+c]*psikz;
            }
          }

          // Fetch the deterministic element residual
          double scale = pc->basis(i,zq)*pc->basis(j,zq)*wq;
          this->delem->addJacobian(elemIndex,
                                   time,
                                   scale*alpha, scale*beta, scale*gamma,
                                   X, uq, udq, uddq,
                                   resq,
                                   A);
        } // quadrature

        // Store (i,j)-th projected Jacobian into stochastic Jacobian
        int iptr = i*nddof;
        int jptr = j*nddof;
        for (int ii = 0; ii < nddof; ii++){
          for (int jj = 0; jj < nddof; jj++){
            addElement(mat, nsdof, iptr + ii, jptr + jj,
                       getElement(A, nddof, ii, jj));
          }
        }

      } // nonzero

    } // end j

  } // end i

  // printSparsity(mat, nddof*nsterms);

  // clear the heap
  delete [] A;
  delete [] resq;
  delete [] uq;
  delete [] udq;
  delete [] uddq;
  delete [] zq;
  delete [] yq;
}
