#include"ParameterContainer.h"
#include"ParameterFactory.h"

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

  // Set max degrees of expansion and get corresponding number of
  // quadrature points
  const int nvars = pc->getNumParameters();
  // int *pmax = new int[nvars];
  // int *nqpts = new int[nvars];
  // for (int i = 0; i < nvars; i++){
  //   pmax[i] = i+2;
  //   nqpts[i] = pmax[i]+1;
  // }


  int pmax[] = {3,3,4,4,2};
  int nqpts[] = {4,4,5,5,3};
  
  printf("max orders        = ");
  for (int i = 0; i < nvars; i++){
    printf("%d ", pmax[i]);
  }
  printf("\nquadrature points = ");
  for (int i = 0; i < nvars; i++){
    printf("%d ", nqpts[i]);
  }
  printf("\n");

  // Initialize basis
  pc->initializeBasis(pmax);
  int nbasis = pc->getNumBasisTerms();

  // Initialize quadrature
  pc->initializeQuadrature(nqpts);  
  int nqpoints = pc->getNumQuadraturePoints();
  
  // int degs[nvars];
  // for (int k = 0; k < nbasis; k++){
  //   printf("\n");
  //   pc->getBasisParamDeg(k, degs);  
  //   for (int i = 0; i < nvars; i++){
  //     printf("%d ", degs[i]);
  //   }
  // }

  printf("%d %d \n", nqpoints, nbasis);
  
  // Space for quadrature points and weights
  double *zq = new double[nvars];
  double *yq = new double[nvars];
  double wq;

  // for (int q = 0; q < nqpoints; q++){
  //   pc->quadrature(q, zq, yq, &wq);
  // }

  for (int k = 0; k < nbasis; k++){
    for (int q = 0; q < nqpoints; q++){
      pc->quadrature(q, zq, yq, &wq);
      pc->basis(k,zq);
      // printf("%6d %6d %13.6f\n", k, q, pc->basis(k,zq));
    }
  }
    
  return 1;
}
