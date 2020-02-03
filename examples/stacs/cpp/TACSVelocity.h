#ifndef TACS_VELOCITY_H
#define TACS_VELOCITY_H

/*
  Compute the structural mass
*/

#include "TACSFunction.h"

/*
  Evaluate the structural mass of the structure
*/
class TACSVelocity : public TACSFunction {
 public:
  TACSVelocity( TACSAssembler * _assembler );
  ~TACSVelocity();

  const char *getObjectName();

  /**
     Member functions to integrate the function value
  */
  void initEvaluation( EvaluationType ftype );
  void elementWiseEval( EvaluationType ftype,
                        int elemIndex, TACSElement *element,
                        double time, TacsScalar scale,
                        const TacsScalar Xpts[], const TacsScalar vars[],
                        const TacsScalar dvars[], const TacsScalar ddvars[] );
  void finalEvaluation( EvaluationType ftype );

  /**
     Return the value of the function
  */
  TacsScalar getFunctionValue();

  /**
     Add the derivative of the function w.r.t. the design variables
  */
  void addElementDVSens( int elemIndex, TACSElement *element,
                         double time, TacsScalar scale,
                         const TacsScalar Xpts[], const TacsScalar vars[],
                         const TacsScalar dvars[], const TacsScalar ddvars[],
                         int dvLen, TacsScalar dfdx[] );

  /**
     Evaluate the derivative of the function w.r.t. the node locations
  */
  void getElementXptSens( int elemIndex, TACSElement *element,
                          double time, TacsScalar scale,
                          const TacsScalar Xpts[], const TacsScalar vars[],
                          const TacsScalar dvars[], const TacsScalar ddvars[],
                          TacsScalar fXptSens[] );

 private:
  // The total mass of all elements in the specified domain
  TacsScalar fval;

  static const char *funcName;
};

#endif // TACS_ENERGY
