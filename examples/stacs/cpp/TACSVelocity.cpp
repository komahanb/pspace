#include "TACSVelocity.h"
#include "TACSAssembler.h"
#include "../smd/smd.h"

/*
  Allocate the structural mass TACSVelocity
*/
TACSVelocity::TACSVelocity( TACSAssembler *_assembler ):
  TACSFunction(_assembler){
  fval = 0.0;
}

/*
  Destructor for the structural mass
*/
TACSVelocity::~TACSVelocity(){}

const char *TACSVelocity::funcName = "Energy";

/*
  The structural mass function name
*/
const char* TACSVelocity::getObjectName(){
  return funcName;
}

/*
  Get the function name
*/
TacsScalar TACSVelocity::getFunctionValue(){
  return fval;
}

/*
  Initialize the mass to zero
*/
void TACSVelocity::initEvaluation( EvaluationType ftype ){
  fval = 0.0;
}

/*
  Sum the mass across all MPI processes
*/
void TACSVelocity::finalEvaluation( EvaluationType ftype ){
  TacsScalar temp = fval;
  MPI_Allreduce(&temp, &fval, 1, TACS_MPI_TYPE,
                MPI_SUM, assembler->getMPIComm());
}

/*
  Perform the element-wise evaluation of the TACSVelocity function.
*/
void TACSVelocity::elementWiseEval( EvaluationType ftype,
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
                                         TACS_VELOCITY_FUNCTION,
                                         time, N, pt,
                                         Xpts, vars, dvars, ddvars,
                                         &quantity);
  fval += scale*quantity;
}

/*
  Determine the derivative of the mass w.r.t. the element nodal
  locations.
*/
void TACSVelocity::getElementXptSens( int elemIndex,
                                           TACSElement *element,
                                           double time,
                                           TacsScalar scale,
                                           const TacsScalar Xpts[],
                                           const TacsScalar vars[],
                                           const TacsScalar dvars[],
                                           const TacsScalar ddvars[],
                                          TacsScalar dfdXpts[] ){}

/*
  Determine the derivative of the mass w.r.t. the material
  design variables
*/
void TACSVelocity::addElementDVSens( int elemIndex,
                                          TACSElement *element,
                                          double time,
                                          TacsScalar scale,
                                          const TacsScalar Xpts[],
                                          const TacsScalar vars[],
                                          const TacsScalar dvars[],
                                          const TacsScalar ddvars[],
                                          int dvLen, TacsScalar dfdx[] ){}
