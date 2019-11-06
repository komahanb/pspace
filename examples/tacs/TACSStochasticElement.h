#ifndef TACS_STOCHASTIC_ELEMENT
#define TACS_STOCHASTIC_ELEMENT

#include "TACSElement.h"
#include "ParameterContainer.h"

class TACSStochasticElement : public TACSElement {
 public:
  TACSStochasticElement( TACSElement *_delem,
                         ParameterContainer *_pc,
                         void (*_update)(TACSElement*, TacsScalar*) );
  ~TACSStochasticElement();

  // TACS Element member functions
  // -----------------------------
  int getVarsPerNode();
  int getNumNodes();

  // Get the element basis
  //-----------------------
  TACSElementBasis* getElementBasis(){
    return delem->getElementBasis();
  }

  int getMultiplierIndex(){
    return delem->getMultiplierIndex();
  }


  // Return the Initial conditions
  // -----------------------------
  void getInitConditions( int elemIndex, const TacsScalar X[],
                          TacsScalar v[], TacsScalar dv[], TacsScalar ddv[] );

  // Compute the residual of the governing equations
  // -----------------------------------------------
  void addResidual( int elemIndex, double time,
                    const TacsScalar X[], const TacsScalar v[],
                    const TacsScalar dv[], const TacsScalar ddv[],
                    TacsScalar res[] );

  // Compute the Jacobian of the governing equations
  // -----------------------------------------------
  void addJacobian( int elemIndex, double time,
                    TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                    const TacsScalar X[], const TacsScalar v[],
                    const TacsScalar dv[], const TacsScalar ddv[],
                    TacsScalar res[], TacsScalar mat[] );

  // Invoke this function to update this element through user supplied callback
  void updateElement(TACSElement* elem, TacsScalar* vals){
    if (this->update != NULL){
      this->update(elem, vals);
    }
  }

 protected:
  TACSElement *delem;
  ParameterContainer *pc;

 private:
  // Stochastic element information
  int num_nodes;
  int vars_per_node;

  // Callback function to update the parameters of element
  void (*update)(TACSElement*, TacsScalar*);
};

#endif
