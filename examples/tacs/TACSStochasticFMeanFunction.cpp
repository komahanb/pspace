#include "TACSAssembler.h"
#include "TACSStochasticFMeanFunction.h"
#include "TACSStochasticElement.h"

namespace {

  void getDeterministicAdjoint( ParameterContainer *pc, 
                                TACSElement *delem,
                                TACSElement *selem, 
                                const TacsScalar v[],
                                double *zq,
                                TacsScalar *uq
                                ){
    int ndvpn   = delem->getVarsPerNode();
    int nsvpn   = selem->getVarsPerNode();
    int nddof   = delem->getNumVariables();
    int nsdof   = selem->getNumVariables();
    int nsterms = pc->getNumBasisTerms();
    int nnodes  = selem->getNumNodes();
    
    memset(uq  , 0, nddof*sizeof(TacsScalar));

    // Evaluate the basis at quadrature node and form the state
    // vectors
    for (int n = 0; n < nnodes; n++){
      for (int k = 0; k < nsterms; k++){
        double psikz = pc->basis(k,zq);
        int lptr = n*ndvpn;
        int gptr = n*nsvpn + k*ndvpn;
        for (int d = 0; d < ndvpn; d++){        
          uq[lptr+d] += v[gptr+d]*psikz;
        }
      }
    }
  } 

  void getDeterministicStates( ParameterContainer *pc, 
                               TACSElement *delem,
                               TACSElement *selem, 
                               const TacsScalar v[],
                               const TacsScalar dv[],
                               const TacsScalar ddv[], 
                               double *zq,
                               TacsScalar *uq,
                               TacsScalar *udq,
                               TacsScalar *uddq
                               ){
    int ndvpn   = delem->getVarsPerNode();
    int nsvpn   = selem->getVarsPerNode();
    int nddof   = delem->getNumVariables();
    int nsdof   = selem->getNumVariables();
    int nsterms = pc->getNumBasisTerms();
    int nnodes  = selem->getNumNodes();

    memset(uq  , 0, nddof*sizeof(TacsScalar));
    memset(udq , 0, nddof*sizeof(TacsScalar));
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
  } 
}

TACSStochasticFMeanFunction::TACSStochasticFMeanFunction( TACSAssembler *tacs,
                                                          TACSFunction *dfunc, 
                                                          ParameterContainer *pc,
                                                          int quantityType,
                                                          int moment_type) 
  : TACSFunction(tacs,
                 dfunc->getDomainType(),
                 dfunc->getStageType(),
                 0)
{  
  this->tacs_comm = tacs->getMPIComm();
  this->dfunc = dfunc;
  this->dfunc->incref();
  this->pc = pc;
  this->quantityType = quantityType;
  this->nsqpts  = pc->getNumQuadraturePoints();
  this->nsterms = pc->getNumBasisTerms();
  this->fvals = new TacsScalar[this->nsterms*this->nsqpts];
  this->moment_type = moment_type;
}

TACSStochasticFMeanFunction::~TACSStochasticFMeanFunction()
{
  this->dfunc->decref();
  this->dfunc = NULL;  
  this->pc = NULL;
  delete [] this->fvals;
}

void TACSStochasticFMeanFunction::initEvaluation( EvaluationType ftype )
{
  memset(this->fvals, 0 , this->nsterms*this->nsqpts*sizeof(TacsScalar));
}

void TACSStochasticFMeanFunction::finalEvaluation( EvaluationType evalType )
{
  TacsScalar temp;
  for (int q = 0; q < nsterms*nsqpts; q++){
    temp = fvals[q];
    MPI_Allreduce(&temp, &fvals[q], 1, TACS_MPI_TYPE, MPI_SUM, this->tacs_comm);
  }
  // Finish up stochastic integration
  const int nsparams = pc->getNumParameters();
  double *zq = new double[nsparams];
  double *yq = new double[nsparams];
  double wq;
  for (int k = 0; k < nsterms; k++){
    for (int q = 0; q < nsqpts; q++){
      double wq = pc->quadrature(q, zq, yq);
      fvals[k*nsqpts+q] *= wq*pc->basis(k,zq);
    }
  }
  delete [] zq;
  delete [] yq;
}

TacsScalar TACSStochasticFMeanFunction::getFunctionValue(){
  return getExpectation();
}

TacsScalar TACSStochasticFMeanFunction::getExpectation(){
  TacsScalar fmean = 0.0;
  for (int k = 0; k < 1; k++){
    for (int q = 0; q < nsqpts; q++){      
      fmean += fvals[k*nsqpts+q];
    }
  }
  return fmean;
}

TacsScalar TACSStochasticFMeanFunction::getVariance(){
  TacsScalar fvar = 0.0;
  for (int k = 1; k < nsterms; k++){
    TacsScalar fk = 0.0;
    for (int q = 0; q < nsqpts; q++){
      fk += fvals[k*nsqpts+q];
    }
    fvar += fk*fk;
  }
  return fvar;
}

void TACSStochasticFMeanFunction::elementWiseEval( EvaluationType evalType,
                                                   int elemIndex,
                                                   TACSElement *element,
                                                   double time,
                                                   TacsScalar tscale,
                                                   const TacsScalar Xpts[],
                                                   const TacsScalar v[],
                                                   const TacsScalar dv[],
                                                   const TacsScalar ddv[] )
{
  TACSStochasticElement *selem = dynamic_cast<TACSStochasticElement*>(element);
  if (!selem) {
    printf("Casting to stochastic element failed; skipping elemenwiseEval");
  };
  
  TACSElement *delem = selem->getDeterministicElement();
  const int nsterms  = pc->getNumBasisTerms();
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
  for (int j = 0; j < nsterms; j++){

    for (int q = 0; q < nsqpts; q++){

      // Get the quadrature points and weights for mean
      wq = pc->quadrature(q, zq, yq);
      
      // Set the parameter values into the element
      selem->updateElement(delem, yq);

      // Form the state vectors
      getDeterministicStates(pc, delem, selem, 
                             v, dv, ddv, zq, 
                             uq, udq, uddq);

      { 
        // spatial and temporal integration
        double pt[3] = {0.0,0.0,0.0};
        int N = 1;
        TacsScalar value = 0.0;
        int count = delem->evalPointQuantity(elemIndex, 
                                             this->quantityType,
                                             time, N, pt,
                                             Xpts, uq, udq, uddq,
                                             &value);
        fvals[j*nsqpts+q] += tscale*value;
      }
 
    } // yq

  } // nsterms

  // clear allocated heap
  delete [] zq;
  delete [] yq;
  delete [] uq;
  delete [] udq;
  delete [] uddq;
}

void TACSStochasticFMeanFunction::getElementSVSens( int elemIndex, TACSElement *element,
                                                    double time,
                                                    TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                                                    const TacsScalar Xpts[],
                                                    const TacsScalar v[],
                                                    const TacsScalar dv[],
                                                    const TacsScalar ddv[],
                                                    TacsScalar dfdu[] )
{
  TACSStochasticElement *selem = dynamic_cast<TACSStochasticElement*>(element);
  if (!selem) {
    printf("Casting to stochastic element failed; skipping elemenwiseEval");
  };

  int numVars = element->getNumVariables();
  memset(dfdu, 0, numVars*sizeof(TacsScalar));
  
  TACSElement *delem = selem->getDeterministicElement();
  const int nsterms  = pc->getNumBasisTerms();
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
    for (int q = 0; q < nsqpts; q++){

      // Get the quadrature points and weights for mean
      wq = pc->quadrature(q, zq, yq);
      double wt = pc->basis(j,zq)*wq;
    
      // Set the parameter values into the element
      selem->updateElement(delem, yq);

      // Form the state vectors
      getDeterministicStates(pc, delem, selem, 
                             v, dv, ddv, zq, 
                             uq, udq, uddq);

      { 
        double pt[3] = {0.0,0.0,0.0};
        int N = 1;
        TacsScalar _dfdq = 1.0;      
        delem->addPointQuantitySVSens(elemIndex,
                                      this->quantityType,
                                      time, wt*alpha, wt*beta, wt*gamma,
                                      N, pt,
                                      Xpts, uq, udq, uddq, &_dfdq, 
                                      dfduj); // store into tmp
      }

    } // end yloop
    
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
  delete [] dfduj;
  delete [] zq;
  delete [] yq;
  delete [] uq;
  delete [] udq;
  delete [] uddq;
}

void TACSStochasticFMeanFunction::addElementDVSens( int elemIndex, TACSElement *element,
                                                    double time, TacsScalar scale,
                                                    const TacsScalar Xpts[], const TacsScalar v[],
                                                    const TacsScalar dv[], const TacsScalar ddv[],
                                                    int dvLen, TacsScalar dfdx[] )
{
  TACSStochasticElement *selem = dynamic_cast<TACSStochasticElement*>(element);
  if (!selem) {
    printf("Casting to stochastic element failed; skipping elemenwiseEval");
  };
  TACSElement *delem = selem->getDeterministicElement();

  const int nsterms  = pc->getNumBasisTerms();
  const int nsparams = pc->getNumParameters();
  const int ndvpn    = delem->getVarsPerNode();
  const int nsvpn    = selem->getVarsPerNode();
  const int nddof    = delem->getNumVariables();
  const int nnodes   = selem->getNumNodes();  
  const int dvpernode = delem->getDesignVarsPerNode();

  // j-th projection of dfdx array
  TacsScalar *dfdxj  = new TacsScalar[dvLen];
  
  // Space for quadrature points and weights
  double *zq = new double[nsparams];
  double *yq = new double[nsparams];
  double wq;
  
  // Create space for deterministic states at each quadrature node in y
  TacsScalar *uq     = new TacsScalar[nddof];
  TacsScalar *udq    = new TacsScalar[nddof];
  TacsScalar *uddq   = new TacsScalar[nddof];
  
  // int nterms;
  // if (moment_type == 0){
  //   nterms = 1;
  // } else {
  //   nterms = 
  // }

  for (int j = 0; j < 1; j++){ 

    memset(dfdxj, 0, dvLen*sizeof(TacsScalar));
    
    // Stochastic Integration
    for (int q = 0; q < nsqpts; q++){

      // Get the quadrature points and weights for mean
      wq = pc->quadrature(q, zq, yq);
      double wt = pc->basis(j,zq)*wq;
    
      // Set the parameter values into the element
      selem->updateElement(delem, yq);

      // form deterministic states      
      getDeterministicStates(pc, delem, selem, v, dv, ddv, zq, uq, udq, uddq);

      // Call the underlying element and get the state variable sensitivities
      double pt[3] = {0.0,0.0,0.0};
      int N = 1;
      TacsScalar _dfdq = 1.0; 
      delem->addPointQuantityDVSens( elemIndex, 
                                     this->quantityType,
                                     time, wt*scale,
                                     N, pt,
                                     Xpts, uq, udq, uddq, &_dfdq, 
                                     dvLen, dfdxj ); 
    } // end yloop

    // for (int n = 0; n < nnodes; n++){
    //   int lptr = n*dvpernode;
    //   int gptr = n*dvpernode + j*dvpernode;
    //   for (int i = 0; i < dvpernode; i++){
    //     dfdx[gptr+n] += dfdxj[lptr+n];
    //   }            
    // }

    // need to be careful with nodewise placement of dvsx
    for (int n = 0; n < dvLen; n++){
      //      printf("term %d dfdx[%d] = %e %e \n", j, n, dfdx[n], dfdxj[n]);
      dfdx[n] += dfdxj[n];
    }
    
  } // end nsterms

  // clear allocated heap
  delete [] zq;
  delete [] yq;
  delete [] uq;
  delete [] udq;
  delete [] uddq;
  delete [] dfdxj;
}
