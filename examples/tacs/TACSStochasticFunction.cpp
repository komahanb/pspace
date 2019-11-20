#include "TACSStochasticFunction.h"
#include "TACSStochasticElement.h"

TACSStochasticFunction::TACSStochasticFunction( TACSAssembler *tacs,
                                                TACSFunction *dfunc, 
                                                ParameterContainer *pc,
                                                int quantityType ) 
  : TACSFunction(tacs,
                 dfunc->getDomainType(),
                 dfunc->getStageType(),
                 0)
{
  this->dfunc = dfunc;
  this->dfunc->incref();
  this->pc = pc;
  this->quantityType = quantityType;
}

TACSStochasticFunction::~TACSStochasticFunction(){
  this->dfunc->decref();
  this->dfunc = NULL;  
  this->pc = NULL;
}

void TACSStochasticFunction::initEvaluation( EvaluationType ftype ){
  this->dfunc->initEvaluation(ftype);
}

void TACSStochasticFunction::finalEvaluation( EvaluationType evalType ){
  this->dfunc->finalEvaluation(evalType);
}

TacsScalar TACSStochasticFunction::getFunctionValue() {
  return this->dfunc->getFunctionValue();
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

    // Call Deterministic function with modified time weight
    double scale = wt*tscale;
    this->dfunc->elementWiseEval(evalType, elemIndex, delem,
                                 time, scale,
                                 Xpts, uq, udq, uddq);    
  } // end yloop

  // clear allocated heap
  delete [] zq;
  delete [] yq;
  delete [] uq;
  delete [] udq;
  delete [] uddq;
}

void TACSStochasticFunction::getElementSVSens( int elemIndex, TACSElement *element,
                                               double time,
                                               TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                                               const TacsScalar Xpts[],
                                               const TacsScalar v[],
                                               const TacsScalar dv[],
                                               const TacsScalar ddv[],
                                               TacsScalar dfdu[] ){
  // zero the values
  int numVars = element->getNumVariables();
  memset(dfdu, 0, numVars*sizeof(TacsScalar));

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

  // j-th project
  TacsScalar *dfduj  = new TacsScalar[nddof];  
  
  // Space for quadrature points and weights
  double *zq = new double[nsparams];
  double *yq = new double[nsparams];
  double wq;
  
  // Create space for deterministic states at each quadrature node in y
  TacsScalar *uq     = new TacsScalar[nddof];
  TacsScalar *udq    = new TacsScalar[nddof];
  TacsScalar *uddq   = new TacsScalar[nddof];

  for (int j = 0; j < nsterms; j++){

    memset(dfduj, 0, nddof*sizeof(TacsScalar));
    
    // Stochastic Integration
    for (int q = 0; q < nqpts; q++){

      // Get the quadrature points and weights for mean
      wq = pc->quadrature(q, zq, yq);
      double wt = pc->basis(j,zq)*wq;
    
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

      // Call Deterministic function with modified time weight
      // this->dfunc->elementWiseEval(evalType, elemIndex, delem,
      //                              time, scale,
      //                              Xpts, uq, udq, uddq);
  
      // Call the underlying element and get the state variable sensitivities
      double pt[3] = {0.0,0.0,0.0};
      int N = 1;
      TacsScalar _dfdq = 1.0;      
      delem->addPointQuantitySVSens(elemIndex,
                                    this->quantityType,
                                    time, wt*alpha, wt*beta, wt*gamma,
                                    N, pt,
                                    Xpts, uq, udq, uddq, &_dfdq, 
                                    dfduj); // store into tmp
    } // end yloop

    for (int n = 0; n < nddof; n++){
      printf("term %d dfdq[%d]=%e\n", j, n, dfduj[n]);
    }
    
    // Store j-th projected sv sens into stochastic array
    for (int n = 0; n < nnodes; n++){
      int lptr = n*ndvpn;
      int gptr = n*nsvpn + j*ndvpn;
      for (int d = 0; d < ndvpn; d++){        
        dfdu[gptr+d] = dfduj[lptr+d];
      }
    }

  } // end nsterms

  // clear allocated heap
  delete [] zq;
  delete [] yq;
  delete [] uq;
  delete [] udq;
  delete [] uddq;
}
