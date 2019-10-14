#include"ParameterContainer.h"
#include"ParameterFactory.h"

ParameterContainer::ParameterContainer(){
  this->tnum_parameters = 0;
}
ParameterContainer::~ParameterContainer(){
  if(param_max_degree){delete [] param_max_degree;};
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

  // Generate and store a set of indices
  this->bhelper->basisDegrees(nvars, pmax, 
                              &this->tnum_basis_terms, 
                              this->dindex);
  }

void  ParameterContainer::initializeQuadrature(const int *nqpts){
  
  const int nvars = getNumParameters();

  // Allocate space for return variables
  int totquadpts = 1;
  for (int i = 0; i < nvars; i++){
    totquadpts *= nqpts[i];
  }

  Z = new double*[nvars];
  Y = new double*[nvars]; 
  for (int i = 0; i < nvars; i++){
    Z[i] = new double[totquadpts];
    Y[i] = new double[totquadpts];    
  }
  W = new double[totquadpts];  

  // Find tensor product of 1d rules
  qh->tensorProduct(nvars, nqpts, zp, yp, wp,
                    Z, Y, W);

}

int main( int argc, char *argv[] ){

  // Create random parameters
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *p1 = factory->createNormalParameter(-4.0, 0.5);
  AbstractParameter *p2 = factory->createUniformParameter(-5.0, 4.0);
  AbstractParameter *p3 = factory->createExponentialParameter(6.0, 1.0);
  AbstractParameter *p4 = factory->createExponentialParameter(6.0, 1.0);
  AbstractParameter *p5 = factory->createNormalParameter(-4.0, 0.5);
   
  // Create container and add random paramters
  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(p1);
  pc->addParameter(p2);
  pc->addParameter(p3);
  pc->addParameter(p4);
  pc->addParameter(p5);

  const int nvars = pc->getNumParameters();
  int *pmax = new int[nvars];
  int *nqpts =new int[nvars];
  for (int i = 0; i < nvars; i++){
    pmax[i] = i+2;
    nqpts[i] = pmax[i]+1;
  }

  printf("max orders        = ");
  for (int i = 0; i < nvars; i++){
    printf("%d ", pmax[i]);
  }
  printf("\nquadrature points = ");
  for (int i = 0; i < nvars; i++){
    printf("%d ", nqpts[i]);
  }
  printf("\n");

  pc->initializeBasis(pmax);

  printf("%d\n", pc->getNumBasisTerms());
  
  for (int k = 0; k < pc->getNumBasisTerms(); k++){
    // pc->initializeQuadrature(nqpts);
    //   printf("\n");
    //   for (int q = 0; q < pc->getNumBasisTerms(); q++){
    //     //pc->quadrature(q, zq, yq, wq);
    //     //printf("%e ", pc->basis(k,zq));
    //   }
  }
    
  return 1;
}
