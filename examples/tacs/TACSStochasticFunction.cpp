#include "TACSStochasticFunction.h"
#include "TACSAssembler.h"
#include "TACSStochasticElement.h"
#include "smd.h"

TACSStochasticFunction::TACSStochasticFunction( TACSAssembler *tacs,
                                                int quantityType,
                                                double ksWeight,
                                                TACSFunction *dfunc, 
                                                ParameterContainer *pc ) 
  : TACSFunction(tacs){
  // Store the deterministic function
  this->dfunc = dfunc;
  this->dfunc->incref();

  this->quantityType = quantityType;
  this->ksWeight = ksWeight;

  // Store the pointer to parameter container
  this->pc = pc;

  // Get the TACS communicator
  this->tacs_comm = tacs->getMPIComm();

  // Allocate space for function values
  //  int nsterms = pc->getNumBasisTerms();
  //  this->fval = new TacsScalar[nsterms];
  //  memset(fval, 0, nsterms*sizeof(TacsScalar));
}

TACSStochasticFunction::~TACSStochasticFunction(){
  this->dfunc->decref();
  this->dfunc = NULL;
  
  this->pc = NULL;
  
  //delete [] this->fval;  
  // this->fval = NULL;
}

void TACSStochasticFunction::initEvaluation( EvaluationType ftype ){
  if (ftype == TACSFunction::INITIALIZE){
    maxValue = -1e20;
  }
  else if (ftype == TACSFunction::INTEGRATE){
    ksSum = 0.0;
  }
}

void TACSStochasticFunction::elementWiseEval( EvaluationType evalType,
                                              int elemIndex,
                                              TACSElement *element,
                                              double time,
                                              TacsScalar tscale,
                                              const TacsScalar Xpts[],
                                              const TacsScalar v[],
                                              const TacsScalar dv[],
                                              const TacsScalar ddv[] ){
  // Access evaluate point quatity of deterministic element
  TACSStochasticElement *selem = dynamic_cast<TACSStochasticElement*>(element);
  if (!selem){    
    printf("Casting failed \n");
    exit(-1);
  }
  TACSElement *delem = selem->getDeterministicElement();

  const int nsterms  = pc->getNumBasisTerms();
  const int nqpts    = pc->getNumQuadraturePoints();
  const int nsparams = pc->getNumParameters();
  const int ndvpn    = delem->getVarsPerNode();
  const int nsvpn    = selem->getVarsPerNode();
  const int nddof    = delem->getNumVariables();
  const int nnodes   = selem->getNumNodes();  
  
  // Space for quadrature points and weights
  double *zq = new double[nsparams];
  double *yq = new double[nsparams];
  double wq;
  
  // Create space for deterministic states at each quadrature node in y
  TacsScalar *uq     = new TacsScalar[nddof];
  TacsScalar *udq    = new TacsScalar[nddof];
  TacsScalar *uddq   = new TacsScalar[nddof];
  
  // Stochastic Integration
  for (int q = 0; q < nqpts; q++){

    // Get the quadrature points and weights
    wq = pc->quadrature(q, zq, yq);
    double wt = pc->basis(0,zq)*wq;
    
    // Set the parameter values into the element
    selem->updateElement(delem, yq);

    // reset the states and residuals
    memset(uq   , 0, nddof*sizeof(TacsScalar));
    memset(udq  , 0, nddof*sizeof(TacsScalar));
    memset(uddq , 0, nddof*sizeof(TacsScalar));
    
    // Evaluate the basis at quadrature node and form the state
    // vectors
    for (int n = 0; n < nnodes; n++){
      for (int k = 0; k < nsterms; k++){
        double psikz = pc->basis(k,zq);
        int lptr = n*ndvpn;
        int gptr = n*nsvpn + k*ndvpn;
        for (int d = 0; d < ndvpn; d++){        
          uq[lptr+d] += v[gptr+d]*psikz;
          udq[lptr+d] += dv[gptr+d]*psikz;
          uddq[lptr+d] += ddv[gptr+d]*psikz;
        }
      }
    }
    
    // Spatial integration
    // Get the number of quadrature points for this element
    const int numGauss = 1; //delem->getNumGaussPts();
    const int numDisps = delem->getNumVariables();
    const int numNodes = delem->getNumNodes();
  
    for ( int i = 0; i < numGauss; i++ ){
    
      // Get the Gauss points one at a time
      double weight = 1.0*wt; //delem->getGaussWtsPts(i, pt);
      double pt[3] = {0.0,0.0,0.0};
      const int n = 1;
      //  delem->getShapeFunctions(pt, ctx->N);
    
      // Evaluate the dot-product with the displacements
      //const double *N = ctx->N;
      const TacsScalar *d = v; //uq[0]; //v;
      TacsScalar energy = 0.0;
      delem->evalPointQuantity(elemIndex,
                               this->quantityType, time, n, pt,
                               Xpts, v, dv, ddv, &energy);        
      TacsScalar value = tscale*energy;
        
      if (evalType == TACSFunction::INITIALIZE){

        printf("initialization \n");
        // Reset maxvalue if needed
        if (TacsRealPart(value) > TacsRealPart(maxValue)){
          printf("Updating maxvalue from %e to %e\n", maxValue, value);
          maxValue = value;
        }

        printf("Skip Updating maxvalue from %e to %e\n", maxValue, value);
              
      } else {
        printf("evaluaation %e %e %e %e\n", value, energy, tscale, ksSum);
        // Add up the contribution from the quadrature
        TacsScalar h = 1.0; //delem->getDetJacobian(pt, Xpts);
        ksSum += h*weight*exp(ksWeight*(value - maxValue));
      }      
    }
    
    // spatial integration

 
  } // Stochastic integration
  
}

void TACSStochasticFunction::finalEvaluation( EvaluationType evalType ){
  if (evalType == TACSFunction::INITIALIZE){
    // Distribute the values of the KS function computed on this domain
    TacsScalar temp = maxValue;
    MPI_Allreduce(&temp, &maxValue, 1, TACS_MPI_TYPE,
                  TACS_MPI_MAX, this->tacs_comm);
  }
  else {
    // Find the sum of the ks contributions from all processes
    TacsScalar temp = ksSum;
    MPI_Allreduce(&temp, &ksSum, 1, TACS_MPI_TYPE,
                  MPI_SUM, this->tacs_comm);
  }
}

/**
   Get the value of the function
*/
TacsScalar TACSStochasticFunction::getFunctionValue() {
  printf("maxvalue = %e weight = %e \n", maxValue, ksWeight);
  return maxValue + log(ksSum)/ksWeight;
}
