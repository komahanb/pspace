#include"ParameterContainer.h"

using namespace std;

ParameterContainer::ParameterContainer(){
  this->tnum_parameters = 0;
}
ParameterContainer::~ParameterContainer(){
  // Clear information about parameters
  if (param_max_degree){ 
    delete [] param_max_degree; 
  };

  // Degree of kth basis entry
  for (int k = 0; k < this->getNumBasisTerms(); k++){
    delete [] this->dindex[k];
  };
  delete [] this->dindex;

  // Deallocate quadrature information
  for (int i = 0; i < this->getNumParameters(); i++){
    delete [] Z[i];
    delete [] Y[i];
  }
  delete [] Z;
  delete [] Y;
  delete [] W;
}

/*
  Add the parameter into  collection
*/
void ParameterContainer::addParameter(AbstractParameter *param){
  if (this->tnum_parameters > 4){
    printf("Warning: Parameter container is full\n");
  }
  this->pmap.insert(std::pair<int, AbstractParameter*>(param->getParameterID(), param));
  this->tnum_parameters++;
}

int ParameterContainer::getNumBasisTerms(){
  return this->tnum_basis_terms;
}

int ParameterContainer::getNumParameters(){
  return this->tnum_parameters;
}

int ParameterContainer::getNumQuadraturePoints(){
  return this->tnum_quadrature_points;
}

void  ParameterContainer::initializeBasis(const int *pmax){
  int nvars = this->getNumParameters();
  
  // Copy over the max degrees
  param_max_degree = new int[nvars];
  for (int k = 0; k < nvars; k++){
    param_max_degree[k] = pmax[k];
  }

  // Number of terms from tensor product
  int nterms = 1;
  for (int i = 0; i < nvars; i++){
    nterms *= 1 + pmax[i];
  }

  // Allocate space for storing degree set
  this->dindex = new int*[nterms];
  for (int k = 0; k < nterms; k++){
    this->dindex[k] = new int[nvars];
  }
  
  // Generate and store a set of indices
  this->bhelper->basisDegrees(nvars, pmax, 
                              &this->tnum_basis_terms, 
                              this->dindex);

  if (this->tnum_basis_terms != nterms){
    printf("Error initializing basis\n");
  };
}

void ParameterContainer::initializeQuadrature(const int *nqpts){  
  const int nvars = getNumParameters();
  int totquadpts = 1;
  for (int i = 0; i < nvars; i++){
    totquadpts *= nqpts[i];
  }
  this->tnum_quadrature_points = totquadpts;
  
  // Get the univariate quadrature points from parameters
  scalar **y = new scalar*[nvars];
  scalar **z = new scalar*[nvars];
  scalar **w = new scalar*[nvars];
  for (int i = 0; i < nvars; i++){
    z[i] = new scalar[nqpts[i]];
    y[i] = new scalar[nqpts[i]];
    w[i] = new scalar[nqpts[i]];    
  }
  map<int,AbstractParameter*>::iterator it;
  for (it = this->pmap.begin(); it != this->pmap.end(); it++){
    int pid = it->first;
    it->second->quadrature(nqpts[pid], z[pid], y[pid], w[pid]);
  }

  // Compute multivariate quadrature and store
  Z = new scalar*[nvars];
  Y = new scalar*[nvars];
  W = new scalar[totquadpts];  
  for (int i = 0; i < nvars; i++){
    Z[i] = new scalar[totquadpts];
    Y[i] = new scalar[totquadpts];    
  }

  // Find tensor product of 1d rules
  qhelper->tensorProduct(nvars, nqpts, z, y, w, Z, Y, W);
  
  // Deallocate space
  for (int i = 0; i < nvars; i++){
    delete [] z[i];
    delete [] y[i];
    delete [] w[i];
  }
  delete [] y;
  delete [] z;
  delete [] w;
}

/*
 */
scalar ParameterContainer::quadrature(int q, scalar *zq, scalar *yq){
  const int nvars = getNumParameters(); 
  for (int i = 0; i < nvars; i++){
    zq[i] = this->Z[i][q];
    yq[i] = this->Y[i][q];
  }
  return this->W[q];
}

/*
  Evaluate the k-the basis function at point "z"
*/
scalar ParameterContainer::basis(int k, scalar *z){
  scalar psi = 1.0;
  map<int,AbstractParameter*>::iterator it;
  for (it = this->pmap.begin(); it != this->pmap.end(); it++){
    int pid = it->first;
    psi *= it->second->basis(z[pid], this->dindex[k][pid]);
  }
  return psi;
}

void ParameterContainer::getBasisParamDeg(int k, int *degs) {
  const int nvars = getNumParameters(); 
  for (int i = 0; i < nvars; i++){
    degs[i] = this->dindex[k][i];
  }
}

void ParameterContainer::getBasisParamMaxDeg(int *pmax) {
  map<int,AbstractParameter*>::iterator it;
  for (it = this->pmap.begin(); it != this->pmap.end(); it++){
    int pid = it->first;
    int dmax = it->second->getMaxDegree();
    pmax[pid] = dmax;
  }
}

/*
  Default initialization 
*/
void ParameterContainer::initialize(){
  const int nvars = getNumParameters();
  int nqpts[nvars];
  int pmax[nvars];    
  map<int,AbstractParameter*>::iterator it;
  for (it = this->pmap.begin(); it != this->pmap.end(); it++){
    int pid = it->first;
    int dmax = it->second->getMaxDegree();
    pmax[pid] = dmax;
    nqpts[pid] = dmax + 1;
  }
  this->initializeBasis(pmax);  
  this->initializeQuadrature(nqpts);
}
