#include "TACSKSStochasticFunction.h"
#include "TACSAssembler.h"
#include "smd.h"
#include "TACSStochasticElement.h"
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

  return;

  TACSStochasticElement *selem = dynamic_cast<TACSStochasticElement*>(element);
  if (!selem) {
    printf("Casting to stochastic element failed; skipping elemenwiseEval");
  };
  
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

    // Get the quadrature points and weights for mean
    wq = pc->quadrature(q, zq, yq);
    double wt = pc->basis(0,zq)*wq;
    
    // Set the parameter values into the element
    selem->updateElement(delem, yq);

    // reset the states and residuals
    memset(uq, 0, nddof*sizeof(TacsScalar));
    memset(udq, 0, nddof*sizeof(TacsScalar));
    memset(uddq, 0, nddof*sizeof(TacsScalar));
    
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

    // Call Deterministic function with modified time weight
    // double scale = wt*tscale;
    // this->dfunc->elementWiseEval(evalType, elemIndex, delem,
    //                              time, scale,
    //                              Xpts, uq, udq, uddq);
    // Find out maxValue[q] if INITIALIZE
    // Find out ksSum[q] if INTEGRATE

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
                                 Xpts, uq, udq, uddq,
                                 &quantity);        
      TacsScalar value = wt*tscale*quantity;
      //      TacsScalar value = quantity;
    
      if (evalType == TACSFunction::INITIALIZE){      
        // Reset maxvalue if needed
        if (TacsRealPart(value) > TacsRealPart(maxValue[q])){
          maxValue[q] = value;
        }      
      } else {
        // Add up the contribution from the quadrature
        //element->getDetJacobian(pt, Xpts);
        TacsScalar h = 1.0;
        ksSum[q] += h*weight*exp(ksWeight*(value - maxValue[q]));
      }      
    }

  } // end yloop

  // clear allocated heap
  delete [] zq;
  delete [] yq;
  delete [] uq;
  delete [] udq;
  delete [] uddq;

}

void TACSKSStochasticFunction::finalEvaluation( EvaluationType evalType )
{
  if (evalType == TACSFunction::INITIALIZE){
    // Distribute the values of the KS function computed on this domain
    TacsScalar temp;
    for (int q = 0; q < nsqpts; q++){
      temp = maxValue[q];
      MPI_Allreduce(&temp, &maxValue[q], 1, TACS_MPI_TYPE, TACS_MPI_MAX, this->tacs_comm);
    }
  }
  else {
    TacsScalar temp;
    for (int q = 0; q < nsqpts; q++){
      temp = ksSum[q];
      MPI_Allreduce(&temp, &ksSum[q], 1, TACS_MPI_TYPE, MPI_SUM, this->tacs_comm);
    }        
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
