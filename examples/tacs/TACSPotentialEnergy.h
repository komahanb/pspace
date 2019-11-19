#ifndef TACS_POTENTIAL_ENERGY_H
#define TACS_POTENTIAL_ENERGY_H

#include "TACSFunction.h"

class TACSPotentialEnergy : public TACSFunction {
 public:
  TACSPotentialEnergy( TACSAssembler * _assembler );
  ~TACSPotentialEnergy();

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
  /*
  void addElementDVSens( int elemIndex, TACSElement *element,
                         double time, TacsScalar scale,
                         const TacsScalar Xpts[], const TacsScalar vars[],
                         const TacsScalar dvars[], const TacsScalar ddvars[],
                         int dvLen, TacsScalar dfdx[] );
  */

  /**
     Evaluate the derivative of the function w.r.t. the node locations
  */
  /*
  void getElementXptSens( int elemIndex, TACSElement *element,
                          double time, TacsScalar scale,
                          const TacsScalar Xpts[], const TacsScalar vars[],
                          const TacsScalar dvars[], const TacsScalar ddvars[],
                          TacsScalar fXptSens[] );
  */
  void getElementSVSens( int elemIndex, TACSElement *element,
                         double time,
                         TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                         const TacsScalar Xpts[],
                         const TacsScalar vars[],
                         const TacsScalar dvars[],
                         const TacsScalar ddvars[],
                         TacsScalar dfdu[] );

 private:
  // The total mass of all elements in the specified domain
  TacsScalar fval;

  static const char *funcName;
};

#endif // TACS_ENERGY
