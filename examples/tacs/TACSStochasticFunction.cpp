#include "TACSStochasticFunction.h"

TACSStochasticFunction::TACSStochasticFunction( TACSAssembler *tacs,
                                                TACSFunction *dfunc, 
                                                ParameterContainer *pc ) 
  : TACSFunction(tacs){
  // Store the deterministic function
  this->dfunc = dfunc;
  this->dfunc->incref();

  // Store the pointer to parameter container
  this->pc = pc;

  // Allocate space for function values
  int nsterms = pc->getNumBasisTerms();
  this->fval = new TacsScalar[nsterms];
  memset(fval, 0, nsterms*sizeof(TacsScalar));
}

TACSStochasticFunction::~TACSStochasticFunction(){
  this->dfunc->decref();
  this->dfunc = NULL;
  this->pc = NULL;
  delete [] this->fval;
  this->fval = NULL;
}


