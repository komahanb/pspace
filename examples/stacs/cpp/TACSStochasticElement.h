#ifndef TACS_STOCHASTIC_ELEMENT
#define TACS_STOCHASTIC_ELEMENT

#include "TACSElement.h"
#include "ParameterContainer.h"
#include "Python.h"

class TACSStochasticElement : public TACSElement {
 public:
  TACSStochasticElement( TACSElement *_delem,
                         ParameterContainer *_pc,
                         void (*_update)(TACSElement*, TacsScalar*, void*) );
  ~TACSStochasticElement();

  void setPythonCallback(PyObject *cbptr){
    printf("setting python callback at address %p", cbptr);
    this->pyptr = cbptr;
  }

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
  /**
    Evaluate a point-wise quantity of interest.
  */
  int evalPointQuantity( int elemIndex, int quantityType, double time,
                         int n, double pt[], const TacsScalar Xpts[],
                         const TacsScalar vars[], const TacsScalar dvars[],
                         const TacsScalar ddvars[], TacsScalar *quantity );
  
  void addAdjResProduct( int elemIndex, double time,
                         TacsScalar scale,
                         const TacsScalar psi[],
                         const TacsScalar Xpts[],
                         const TacsScalar v[],
                         const TacsScalar dv[],
                         const TacsScalar ddv[],
                         int dvLen, 
                         TacsScalar dfdx[] );
 
  // Invoke this function to update this element through user supplied callback
  //---------------------------------------------------------------------------
  void updateElement(TACSElement* elem, TacsScalar* vals){
    if (this->update && pyptr){
      this->update(elem, vals, pyptr);
    } else {
      printf("skipping update of parameters \n");
    }
  }

  TACSElement* getDeterministicElement(){
    return this->delem;
  };
  
  /**
     Get the number of design variables per node.
     
     The value defaults to one, unless over-ridden by the model
  */
  int getDesignVarsPerNode(){
    return this->delem->getDesignVarsPerNode();
  }
  
  /**
     Retrieve the global design variable numbers associated with this element

     Note when the dvNums argument is NULL, then the result is a query
     on the number of design variables and the array is not set.

     @param dvLen The length of the array dvNums
     @param dvNums An array of the design variable numbers for this element
     @return The number of design variable numbers defined by the element
  */
  int getDesignVarNums( int elemIndex, int dvLen, int dvNums[] ){
    return this->delem->getDesignVarNums(elemIndex, dvLen, dvNums);
  }

  /**
     Get the element design variables values

     @param elemIndex The local element index
     @param dvLen The length of the design array
     @param dvs The design variable values
     @return The number of design variable numbers defined by the element
  */
  int getDesignVars( int elemIndex, int dvLen, TacsScalar dvs[] ){
    return this->delem->getDesignVars(elemIndex, dvLen, dvs);
  }

  /**
     Set the element design variables from the design vector

     @param elemIndex The local element index
     @param dvLen The length of the design array
     @param dvs The design variable values
     @return The number of design variable numbers defined by the element
  */
  int setDesignVars( int elemIndex,
                     int dvLen, const TacsScalar dvs[] ){
    return this->delem->setDesignVars(elemIndex, dvLen, dvs);
  }

  /**
     Get the lower and upper bounds for the design variable values

     @param elemIndex The local element index
     @param dvLen The length of the design array
     @param lowerBound The design variable lower bounds
     @param lowerBound The design variable upper bounds
     @return The number of design variable numbers defined by the element
  */
  int getDesignVarRange( int elemIndex, int dvLen,
                         TacsScalar lowerBound[],
                         TacsScalar upperBound[] ){
    return this->delem->getDesignVarRange(elemIndex, dvLen, lowerBound, upperBound);
  }

  // Callback function to update the parameters of element
  void (*update)(TACSElement*, TacsScalar*, void*);
  PyObject *pyptr; 

 protected:
  TACSElement *delem;
  ParameterContainer *pc;

 private:
  // Stochastic element information
  int num_nodes;
  int vars_per_node;
};

#endif
