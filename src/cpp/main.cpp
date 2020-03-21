#include "scalar.h"
#include "ParameterContainer.h"
#include "ParameterFactory.h"

int main( int argc, char *argv[] ){

  // Create random parameterse
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *p1 = factory->createNormalParameter(-4.0, 0.5, 4);
  AbstractParameter *p2 = factory->createUniformParameter(-5.0, 4.0, 3);
  // AbstractParameter *p3 = factory->createExponentialParameter(6.0, 1.0, 6);
  // AbstractParameter *p4 = factory->createUniformParameter(-5.0, 4.0, 4);
  //  AbstractParameter *p5 = factory->createExponentialParameter(6.0, 1.0, 4);

  // Create container and add random paramters
  ParameterContainer *pc = new ParameterContainer(1);
  pc->addParameter(p1);
  pc->addParameter(p2);
  // pc->addParameter(p3);
  //  pc->addParameter(p4);
  // pc->addParameter(p5);

  // int pmax[] = {3,3}; //{2,2,2,2,2};
  // int nqpts[] = {4,4}; //,3,3,3};
  
  // //printf("initializing basis\n");
  // pc->initializeBasis(pmax);

  // //printf("initializing quadrature\n");
  // pc->initializeQuadrature(nqpts);

  pc->initialize();
  
  // Initialize basis
  int nbasis = pc->getNumBasisTerms();
  int nqpoints = pc->getNumQuadraturePoints();

  const int nvars = pc->getNumParameters();
  printf("nbasis %d\n", nbasis);

  int degs[nvars];
  for (int k = 0; k < nbasis; k++){
    printf("%d [ ", k);
    for (int n = 0; n < nvars; n++){
      pc->getBasisParamDeg(k, degs);
      printf("%d ", degs[n]);
    }
    printf("]\n");
  }

  // Space for quadrature points and weights
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

  delete factory;
  delete pc;

  return 1;
}
