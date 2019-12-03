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
    double weight       = 1.0; //element->getGaussWtsPts(i, pt);
    double pt[3]        = {0.0,0.0,0.0};
    const int n         = 1;
    TacsScalar quantity = 0.0;
    element->evalPointQuantity(elemIndex,
                               this->quantityType,
                               time, n, pt,
                               Xpts, v, dv, ddv,
                               &quantity);        
    TacsScalar value = quantity;
    
    if (evalType == TACSFunction::INITIALIZE){      
      if (TacsRealPart(value) > TacsRealPart(maxValue)){
        maxValue = value;
      }
    } else {
      ksSum += tscale*exp(ksWeight*(value - maxValue));
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

void TACSKSFunction::getElementSVSens( int elemIndex, TACSElement *element,
                                       double time,
                                       TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                                       const TacsScalar Xpts[],
                                       const TacsScalar v[],
                                       const TacsScalar dv[],
                                       const TacsScalar ddv[],
                                       TacsScalar dfdu[] ){
  if (RealPart(ksSum) < 1.0e-15){
    printf("Error: Evaluate the functions before derivatives \n");
  }

  // Get the number of quadrature points for this element
  const int numGauss = 1; //element->getNumGaussPts();
  const int numDisps = element->getNumVariables();
  const int numNodes = element->getNumNodes();
  
  memset(dfdu, 0, numDisps*sizeof(TacsScalar));
  
  for ( int i = 0; i < numGauss; i++ ){      
    double weight       = 1.0; //element->getGaussWtsPts(i, pt);
    double pt[3]        = {0.0,0.0,0.0};
    const int n         = 1;

    TacsScalar quantity = 0.0;
    element->evalPointQuantity(elemIndex,
                               this->quantityType,
                               time, n, pt,
                               Xpts, v, dv, ddv,
                               &quantity);        
    
    TacsScalar ksPtWeight = 0.0;
    ksPtWeight = exp(ksWeight*(quantity - maxValue))/ksSum;
    // ksPtWeight *= weight*detJ;

    TacsScalar dfdq = ksPtWeight;
    element->addPointQuantitySVSens(elemIndex,
                                    this->quantityType,
                                    time,
                                    alpha*ksPtWeight, beta*ksPtWeight, gamma*ksPtWeight,
                                    n, pt, Xpts, v, dv, ddv,
                                    &dfdq, dfdu);
  }
}

void TACSKSFunction::addElementDVSens( int elemIndex, TACSElement *element,
                                       double time, TacsScalar scale,
                                       const TacsScalar Xpts[], const TacsScalar v[],
                                       const TacsScalar dv[], const TacsScalar ddv[],
                                       int dvLen, TacsScalar dfdx[] ){
  if (RealPart(ksSum) < 1.0e-15){
    printf("Error: Evaluate the functions before derivatives \n");
  }

  // Get the number of quadrature points for this element
  const int numGauss = 1; //element->getNumGaussPts();
  const int numDisps = element->getNumVariables();
  const int numNodes = element->getNumNodes();
  
  for ( int i = 0; i < numGauss; i++ ){      
    double weight       = 1.0; //element->getGaussWtsPts(i, pt);
    double pt[3]        = {0.0,0.0,0.0};
    const int n         = 1;

    TacsScalar quantity = 0.0;
    element->evalPointQuantity(elemIndex,
                               this->quantityType,
                               time, n, pt,
                               Xpts, v, dv, ddv,
                               &quantity);        

    TacsScalar ksPtWeight = 0.0;
    ksPtWeight = exp(ksWeight*(quantity - maxValue))/ksSum;
    // ksPtWeight *= weight*detJ;

    TacsScalar dfdq = ksPtWeight;
    element->addPointQuantityDVSens(elemIndex,
                                    this->quantityType,
                                    time,
                                    scale*ksPtWeight, n, pt,
                                    Xpts, v, dv, ddv,
                                    &dfdq, dvLen, dfdx);
  }
}

