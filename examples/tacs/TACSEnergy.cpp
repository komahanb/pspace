#include "TACSEnergy.h"
#include "TACSAssembler.h"
#include "smd.h"

/*
  Allocate the structural mass TACSEnergy
*/
TACSEnergy::TACSEnergy( TACSAssembler *_assembler ):
  TACSFunction(_assembler){
  fval = 0.0;
}

/*
  Destructor for the structural mass
*/
TACSEnergy::~TACSEnergy(){}

const char *TACSEnergy::funcName = "Energy";

/*
  The structural mass function name
*/
const char* TACSEnergy::getObjectName(){
  return funcName;
}

/*
  Get the function name
*/
TacsScalar TACSEnergy::getFunctionValue(){
  return fval;
}

/*
  Initialize the mass to zero
*/
void TACSEnergy::initEvaluation( EvaluationType ftype ){
  fval = 0.0;
}

/*
  Sum the mass across all MPI processes
*/
void TACSEnergy::finalEvaluation( EvaluationType ftype ){
  TacsScalar temp = fval;
  MPI_Allreduce(&temp, &fval, 1, TACS_MPI_TYPE,
                MPI_SUM, assembler->getMPIComm());
}

/*
  Perform the element-wise evaluation of the TACSEnergy function.
*/
void TACSEnergy::elementWiseEval( EvaluationType ftype,
                                  int elemIndex,
                                  TACSElement *element,
                                  double time,
                                  TacsScalar scale,
                                  const TacsScalar Xpts[],
                                  const TacsScalar vars[],
                                  const TacsScalar dvars[],
                                  const TacsScalar ddvars[] ){
  // todo check evaluation type is integrate
  TacsScalar energy = 0.0;
  double pt[3] = {0.0,0.0,0.0};
  int N = 1;
  int count = element->evalPointQuantity(elemIndex, 
                                         TACS_KINETIC_ENERGY_FUNCTION,
                                         time, N, pt,
                                         Xpts, vars, dvars, ddvars,
                                         &energy);
  fval += scale*energy;
}

/*
  Determine the derivative of the mass w.r.t. the element nodal
  locations.
*/
void TACSEnergy::getElementXptSens( int elemIndex,
                                    TACSElement *element,
                                    double time,
                                    TacsScalar scale,
                                    const TacsScalar Xpts[],
                                    const TacsScalar vars[],
                                    const TacsScalar dvars[],
                                    const TacsScalar ddvars[],
                                    TacsScalar dfdXpts[] ){
  // Zero the derivative of the function w.r.t. the node locations
  int numNodes = element->getNumNodes();
  memset(dfdXpts, 0, 3*numNodes*sizeof(TacsScalar));

  // Get the element basis class
  TACSElementBasis *basis = element->getElementBasis();

  if (basis){
    for ( int i = 0; i < basis->getNumQuadraturePoints(); i++ ){
      double pt[3];
      double weight = basis->getQuadraturePoint(i, pt);

      TacsScalar density = 0.0;
      int count = element->evalPointQuantity(elemIndex, TACS_ELEMENT_DENSITY,
                                             time, i, pt,
                                             Xpts, vars, dvars, ddvars,
                                             &density);

      if (count >= 1){
        // Evaluate the determinant of the Jacobian
        TacsScalar Xd[9], J[9];
        basis->getJacobianTransform(pt, Xpts, Xd, J);

        // Compute the sensitivity contribution
        TacsScalar dfddetJ = density*weight;
        basis->addJacobianTransformXptSens(pt, Xd, J, scale*dfddetJ,
                                           NULL, NULL, dfdXpts);
      }
    }
  }
}

/*
  Determine the derivative of the mass w.r.t. the material
  design variables
*/
void TACSEnergy::addElementDVSens( int elemIndex,
                                   TACSElement *element,
                                   double time,
                                   TacsScalar scale,
                                   const TacsScalar Xpts[],
                                   const TacsScalar vars[],
                                   const TacsScalar dvars[],
                                   const TacsScalar ddvars[],
                                   int dvLen, TacsScalar dfdx[] ){
  // Get the element basis class
  TACSElementBasis *basis = element->getElementBasis();

  if (basis){
    for ( int i = 0; i < basis->getNumQuadraturePoints(); i++ ){
      double pt[3];
      double weight = basis->getQuadraturePoint(i, pt);

      TacsScalar density = 0.0;
      int count = element->evalPointQuantity(elemIndex, TACS_ELEMENT_DENSITY,
                                             time, i, pt,
                                             Xpts, vars, dvars, ddvars,
                                             &density);
      if (count >= 1){
        // Evaluate the determinant of the Jacobian
        TacsScalar Xd[9], J[9];
        TacsScalar detJ = basis->getJacobianTransform(pt, Xpts, Xd, J);
        TacsScalar dfdq = weight*detJ;

        element->addPointQuantityDVSens(elemIndex, TACS_ELEMENT_DENSITY,
                                        time, scale, i, pt,
                                        Xpts, vars, dvars, ddvars,
                                        &dfdq, dvLen, dfdx);
      }
    }
  }
}
