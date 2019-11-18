#include "TACSKSStochasticFunction.h"
#include "TACSAssembler.h"
#include "smd.h"

TACSKSStochasticFunction::TACSKSStochasticFunction( TACSFunction *dfunc, 
                                                    int quantityType,
                                                    double ksWeight,
                                                    ParameterContainer *pc, 
                                                    int moment_type ) 
  : TACSFunction(dfunc->getAssembler(), 
                 dfunc->getDomainType(), 
                 dfunc->getStageType(),
                 0)
{ 
  // Store pointers 
  this->quantityType = quantityType;
  this->ksWeight = ksWeight;
  this->pc = pc;
  this->moment_type = moment_type; // mean or std. deviation

  // Memory allocation stuff
  this->tacs_comm = dfunc->getAssembler()->getMPIComm();
  this->nsqpts = pc->getNumQuadraturePoints();

  // Create copies of deterministic element
  /*
    this->dfunc = new TACSFunction*[nsqpts];
    for ( int q = 0; q < this->nsqpts; q++){
    dfunc->incref();
    this->dfunc[q] = dfunc;
    }
  */

  this->ksSum = new TacsScalar[nsqpts];
  memset(this->ksSum, 0 , nsqpts*sizeof(TacsScalar*));

  this->maxValue = new TacsScalar[nsqpts];
  memset(this->maxValue, 0 , nsqpts*sizeof(TacsScalar*));
}

TACSKSStochasticFunction::~TACSKSStochasticFunction(){
  delete [] this->ksSum;
  delete [] this->maxValue;
}

void TACSKSStochasticFunction::initEvaluation( EvaluationType ftype )
{
  if (ftype == TACSFunction::INITIALIZE){
    for ( int q = 0; q < this->nsqpts; q++){
      maxValue[q] = -1e20;
    }
  }
  else if (ftype == TACSFunction::INTEGRATE){
    for ( int q = 0; q < this->nsqpts; q++){
      ksSum[q] = 0.0;
    }
  }
}

void TACSKSStochasticFunction::elementWiseEval( EvaluationType evalType,
                                                int elemIndex,
                                                TACSElement *element,
                                                double time,
                                                TacsScalar tscale,
                                                const TacsScalar Xpts[],
                                                const TacsScalar v[],
                                                const TacsScalar dv[],
                                                const TacsScalar ddv[] )
{
  // Find out maxValue[q] if INITIALIZE

  // Find out ksSum[q] if INTEGRATE


/*
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
*/

}

void TACSKSStochasticFunction::finalEvaluation( EvaluationType evalType )
{
  if (evalType == TACSFunction::INITIALIZE){
    // Distribute the values of the KS function computed on this domain
    TacsScalar temp[nsqpts];
    for (int q = 0; q < nsqpts; q++){
      temp[q] = maxValue[q];
    }
    MPI_Allreduce(&temp, &maxValue, nsqpts, TACS_MPI_TYPE,
                  TACS_MPI_MAX, this->tacs_comm);
  }
  else {
    // Find the sum of the ks contributions from all processes
    TacsScalar temp[nsqpts];
    for (int q = 0; q < nsqpts; q++){
      temp[q] = ksSum[q];
    }
    MPI_Allreduce(&temp, &ksSum, nsqpts, TACS_MPI_TYPE,
                  MPI_SUM, this->tacs_comm);
  }
}

/**
   Get the value of the function
*/
TacsScalar TACSKSStochasticFunction::getFunctionValue() {
  TacsScalar fmean = 0.0;
  const int nsparams = pc->getNumParameters();
  double *zq = new double[nsparams];
  double *yq = new double[nsparams];
  double wq;
  for (int k = 0; k < 1; k++){
    for (int q = 0; q < nsqpts; q++){
      double wq = pc->quadrature(q, zq, yq);
      fmean += wq*(maxValue[q] + log(ksSum[q])/ksWeight);
    }
  }
  delete [] zq;
  delete [] yq;
  return fmean;

  //  return maxValue + log(ksSum)/ksWeight;
  /*
  const int nsparams = pc->getNumParameters();
  double *zq = new double[nsparams];
  double *yq = new double[nsparams];
  double wq;

  // Do integration in probabilistic domain finally
  if (moment_type == 0){
    TacsScalar fmean = 0.0;
    for (int k = 0; k < 1; k++){
      for (int q = 0; q < nsqpts; q++){
        wq = pc->quadrature(q, zq, yq);
        fmean += wq*pc->basis(k,zq)*(maxValue[q] + log(ksSum[q])/ksWeight);
      }
    }
    return fmean;
  } else {
    // Compute standard deviation
    TacsScalar fvar = 0.0;
    const int nsterms = pc->getNumBasisTerms();
    for (int k = 1; k < nsterms; k++){
      for (int q = 0; q < nsqpts; q++){
        wq = pc->quadrature(q, zq, yq);
        fvar += wq*pc->basis(k,zq)*(maxValue[q] + log(ksSum[q])/ksWeight);
      }
    }
    return fvar;
  }
*/
}
