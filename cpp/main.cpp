#include"ParameterContainer.h"
#include"ParameterFactory.h"

int main( int argc, char *argv[] ){

  // Create random parameterse
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *p1 = factory->createNormalParameter(-4.0, 0.5, 0);
  AbstractParameter *p2 = factory->createUniformParameter(-5.0, 4.0, 0);
  AbstractParameter *p3 = factory->createExponentialParameter(6.0, 1.0, 0);
  AbstractParameter *p4 = factory->createUniformParameter(-5.0, 4.0, 0);
  AbstractParameter *p5 = factory->createExponentialParameter(6.0, 1.0, 0);

  // Create container and add random paramters
  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(p1);
  pc->addParameter(p2);
  pc->addParameter(p3);
  pc->addParameter(p4);
  pc->addParameter(p5);

  int pmax[] = {2,2,2,2,2};
  int nqpts[] = {3,3,3,3,3};
  
  printf("initializing basis\n");
  pc->initializeBasis(pmax);

  printf("initializing quadrature\n");
  pc->initializeQuadrature(nqpts);
  
  // Initialize basis
  int nbasis = pc->getNumBasisTerms();
  int nqpoints = pc->getNumQuadraturePoints();

  printf("nbasis %d\n ", nbasis);
  
  // Space for quadrature points and weights
  const int nvars = pc->getNumParameters();
  scalar *zq = new scalar[nvars];
  scalar *yq = new scalar[nvars];
  scalar wq;
  for (int k = 0; k < nbasis; k++){
    for (int q = 0; q < nqpoints; q++){
      wq = pc->quadrature(q, zq, yq);
      pc->basis(k,zq);
    }
  }  

  delete [] zq;
  delete [] yq;

  delete p1;
  delete p2;
  delete p3;
  delete p4;
  delete p5;
  delete factory;
  delete pc;

  return 1;
}
