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
  TACSElement *delem = selem->getDeterministicElement();

  // Stochastic Integration
  { 

    // for each quadrature point in Y domain

    //       update the deterministic element
    
     // Spatial integration

    {

      // Get the number of quadrature points for this element
      const int numGauss = 1; //delem->getNumGaussPts();
      const int numDisps = delem->getNumVariables();
      const int numNodes = delem->getNumNodes();
  
      for ( int i = 0; i < numGauss; i++ ){
    
        // Get the Gauss points one at a time
        double weight = 1.0; //delem->getGaussWtsPts(i, pt);
        double pt[3] = {0.0,0.0,0.0};
        const int n = 1;
        //  delem->getShapeFunctions(pt, ctx->N);
    
        // Evaluate the dot-product with the displacements
        //const double *N = ctx->N;
        const TacsScalar *d = v;
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
    
    } // spatial integration

    // use the weights
    
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
