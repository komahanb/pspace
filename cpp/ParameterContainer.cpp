#include"ParameterContainer.h"
#include"ParameterFactory.h"


ParameterContainer::ParameterContainer(){
  this->tnum_parameters = 0;
}
ParameterContainer::~ParameterContainer(){

  if(param_max_degree){delete [] param_max_degree;};

  for (int k = 0; k < this->tnum_basis_terms; k++){
    delete [] this->dindex[k];
  };
  delete [] this->dindex;

  // Deallocate Z, Y, W  
}

/*
  Add the parameter into  collection
*/
void ParameterContainer::addParameter(AbstractParameter *param){
  if (this->tnum_parameters > 4){
    printf("Warning: Parameter container is full\n");
  }
  this->pmap.insert(pair<int, AbstractParameter*>(param->getParameterID(), param));
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
  double **y = new double*[nvars];
  double **z = new double*[nvars];
  double **w = new double*[nvars];
  for (int i = 0; i < nvars; i++){
    z[i] = new double[nqpts[i]];
    y[i] = new double[nqpts[i]];
    w[i] = new double[nqpts[i]];    
  }
  map<int,AbstractParameter*>::iterator it;
  for (it = this->pmap.begin(); it != this->pmap.end(); it++){
    int pid = it->first;
    it->second->quadrature(nqpts[pid], z[pid], y[pid], w[pid]);
  }

  // Allocate space for return variables  
  // for (int i = 0; i < nvars; i++){
  //   if (Z[i]){ delete Z[i]; };
  //   if (Y[i]){ delete Y[i]; };
  // }
  // if (Z){ delete [] Z; };
  // if (Y){ delete [] Y; };
  // if (W){ delete [] W; };  

  Z = new double*[nvars];
  Y = new double*[nvars];
  W = new double[totquadpts];  
  for (int i = 0; i < nvars; i++){
    Z[i] = new double[totquadpts];
    Y[i] = new double[totquadpts];    
  }

  // Find tensor product of 1d rules
  qhelper->tensorProduct(nvars, nqpts, z, y, w, Z, Y, W);
  
  // Deallocate space
  for (int i = 0; i < nvars; i++){
    delete z[i];
    delete y[i];
    delete w[i];
  }
  delete [] y;
  delete [] z;
  delete [] w;
}

/*
 */
void ParameterContainer::quadrature(int q,
                                    double *zq, double *yq, double *wq){
  const int nvars = getNumParameters(); 
  wq[0] = this->W[q];
  for (int i = 0; i < nvars; i++){
    zq[i] = this->Z[i][q];
    yq[i] = this->Y[i][q];
  }
}

/*
  Evaluate the k-the basis function at point "z"
*/
double ParameterContainer::basis(int k, double *z){
  double psi = 1.0;
  map<int,AbstractParameter*>::iterator it;
  for (it = this->pmap.begin(); it != this->pmap.end(); it++){
    int pid = it->first;
    //printf("%d %d \n", pid, this->dindex[k][pid]);
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

/*
  Iterate through the map of parameters and update the state of
  objects

  obj : input element/constitutive obj
  yq : values
*/
void ParameterContainer::updateParameters(int cid, void *obj, const double *yq){
  map<int,AbstractParameter*>::iterator it;
  for (it = this->pmap.begin(); it != this->pmap.end(); it++){
    int pid = it->first;    
    //  printf(" param[%d] = %f ", pid, it->second->getValue(cid, obj));

    
    it->second->updateValue(cid, obj, yq[pid]);
    // printf("param[%d] = %f \n", pid, it->second->getValue(cid, obj));
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
