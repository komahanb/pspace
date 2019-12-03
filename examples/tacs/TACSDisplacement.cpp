#include "TACSDisplacement.h"
#include "TACSAssembler.h"
#include "smd.h"

/*
  Allocate the structural mass TACSDisplacement
*/
TACSDisplacement::TACSDisplacement( TACSAssembler *_assembler ):
  TACSFunction(_assembler){
  fval = 0.0;
}

/*
  Destructor for the structural mass
*/
TACSDisplacement::~TACSDisplacement(){}

const char *TACSDisplacement::funcName = "Energy";

/*
  The structural mass function name
*/
const char* TACSDisplacement::getObjectName(){
  return funcName;
}

/*
  Get the function name
*/
TacsScalar TACSDisplacement::getFunctionValue(){
  return fval;
}

/*
  Initialize the mass to zero
*/
void TACSDisplacement::initEvaluation( EvaluationType ftype ){
  fval = 0.0;
}

/*
  Sum the mass across all MPI processes
*/
void TACSDisplacement::finalEvaluation( EvaluationType ftype ){
  TacsScalar temp = fval;
  MPI_Allreduce(&temp, &fval, 1, TACS_MPI_TYPE,
                MPI_SUM, assembler->getMPIComm());
}

/*
  Perform the element-wise evaluation of the TACSDisplacement function.
*/
void TACSDisplacement::elementWiseEval( EvaluationType ftype,
                                         int elemIndex,
                                         TACSElement *element,
                                         double time,
                                         TacsScalar scale,
                                         const TacsScalar Xpts[],
                                         const TacsScalar vars[],
                                         const TacsScalar dvars[],
                                         const TacsScalar ddvars[] ){
  // todo check evaluation type is integrate
  TacsScalar quantity = 0.0;
  double pt[3] = {0.0,0.0,0.0};
  int N = 1;
  int count = element->evalPointQuantity(elemIndex, 
                                         TACS_DISPLACEMENT_FUNCTION,
                                         time, N, pt,
                                         Xpts, vars, dvars, ddvars,
                                         &quantity);
  fval += scale*quantity;
}

void TACSDisplacement::getElementSVSens( int elemIndex, TACSElement *element,
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

  //Call the underlying element and get the state variable sensitivities
  double pt[3] = {0.0,0.0,0.0};
  int N = 1;
  TacsScalar _dfdq = 1.0;
  element->addPointQuantitySVSens( elemIndex, 
                                   TACS_DISPLACEMENT_FUNCTION,
                                   time, alpha, beta, gamma,
                                   N, pt,
                                   Xpts, v, dv, ddv, &_dfdq, 
                                   dfdu);
  //  printf("displacement dfdu = %.17e\n", dfdu[0]);
}

void TACSDisplacement::addElementDVSens( int elemIndex, TACSElement *element,
                                         double time, TacsScalar scale,
                                         const TacsScalar Xpts[], const TacsScalar v[],
                                         const TacsScalar dv[], const TacsScalar ddv[],
                                         int dvLen, TacsScalar dfdx[] ){
  // printf("entering dv sens in displacment\n");
  double pt[3] = {0.0,0.0,0.0};
  int N = 1;
  TacsScalar _dfdq = 1.0; 
  element->addPointQuantityDVSens( elemIndex, 
                                   TACS_DISPLACEMENT_FUNCTION,
                                   time, scale,
                                   N, pt,
                                   Xpts, v, dv, ddv, &_dfdq, 
                                   dvLen, dfdx );
}
