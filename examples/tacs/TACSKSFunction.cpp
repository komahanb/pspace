#include "TACSKSFunction.h"
#include "TACSAssembler.h"
#include "smd.h"

TACSKSFunction::TACSKSFunction( TACSAssembler *tacs,
                                int quantityType,
                                double ksWeight ) 
  : TACSFunction(tacs, TACSFunction::ENTIRE_DOMAIN, TACSFunction::TWO_STAGE, 0)
{
  this->quantityType = quantityType;
  this->ksWeight = ksWeight;
  this->tacs_comm = tacs->getMPIComm();
  this->ksSum = 0.0;
  this->maxValue = -1e20;
}

TACSKSFunction::~TACSKSFunction(){}

void TACSKSFunction::initEvaluation( EvaluationType ftype )
{
  if (ftype == TACSFunction::INITIALIZE){
    maxValue = -1e20;
  }
  else if (ftype == TACSFunction::INTEGRATE){
    ksSum = 0.0;
  }
}

void TACSKSFunction::elementWiseEval( EvaluationType evalType,
                                      int elemIndex,
                                      TACSElement *element,
                                      double time,
                                      TacsScalar tscale,
                                      const TacsScalar Xpts[],
                                      const TacsScalar v[],
                                      const TacsScalar dv[],
                                      const TacsScalar ddv[] )
{
  // Get the number of quadrature points for this element
  const int numGauss = 1; //element->getNumGaussPts();
  const int numDisps = element->getNumVariables();
  const int numNodes = element->getNumNodes();
  
  for ( int i = 0; i < numGauss; i++ ){
      
    // Get the Gauss points one at a time
    double weight = 1.0; //element->getGaussWtsPts(i, pt);
    double pt[3] = {0.0,0.0,0.0};
    const int n = 1;
    //  element->getShapeFunctions(pt, ctx->N);
    
    // Evaluate the dot-product with the displacements
    //const double *N = ctx->N;
    TacsScalar quantity = 0.0;
    element->evalPointQuantity(elemIndex,
                               this->quantityType,
                               time, n, pt,
                               Xpts, v, dv, ddv,
                               &quantity);        
    //    TacsScalar value = tscale*quantity;
    TacsScalar value = quantity;
    
    if (evalType == TACSFunction::INITIALIZE){      
      // Reset maxvalue if needed
      if (TacsRealPart(value) > TacsRealPart(maxValue)){
        maxValue = value;
      }      
    } else {
      // Add up the contribution from the quadrature
      //element->getDetJacobian(pt, Xpts);
      TacsScalar h = 1.0;
      ksSum += h*weight*exp(ksWeight*(value - maxValue));
    }      
  }
}

void TACSKSFunction::finalEvaluation( EvaluationType evalType )
{
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
TacsScalar TACSKSFunction::getFunctionValue() {
  return maxValue + log(ksSum)/ksWeight;
}
