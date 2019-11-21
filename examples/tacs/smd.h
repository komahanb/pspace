#include "TACSElement.h"

// Define some quantities of interest
static const int TACS_KINETIC_ENERGY_FUNCTION   = -1;
static const int TACS_POTENTIAL_ENERGY_FUNCTION = -2;
static const int TACS_DISPLACEMENT_FUNCTION     = -3;
static const int TACS_VELOCITY_FUNCTION         = -4;

class SMD : public TACSElement{  
 public:
  SMD(double m, double c, double k);

  /**
     Return the Initial conditions
  */
  void getInitConditions( int elemIndex, const TacsScalar X[],
                          TacsScalar v[], TacsScalar dv[], TacsScalar ddv[] );

  /**
     Compute the residual of the governing equations
  */
  void addResidual( int elemIndex, double time,
                    const TacsScalar X[], const TacsScalar v[],
                    const TacsScalar dv[], const TacsScalar ddv[],
                    TacsScalar res[] );

  /**
     Compute the Jacobian of the governing equations
  */
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

  void addPointQuantitySVSens( int elemIndex, int quantityType,
                               double time,
                               TacsScalar alpha,
                               TacsScalar beta,
                               TacsScalar gamma,
                               int n, double pt[],
                               const TacsScalar Xpts[],
                               const TacsScalar vars[],
                               const TacsScalar dvars[],
                               const TacsScalar ddvars[],
                               const TacsScalar dfdq[],
                               TacsScalar dfdu[] );

  void addPointQuantityDVSens( int elemIndex, int quantityType,
                               double time,
                               TacsScalar scale,
                               int n, double pt[],
                               const TacsScalar Xpts[],
                               const TacsScalar vars[],
                               const TacsScalar dvars[],
                               const TacsScalar ddvars[],
                               const TacsScalar dfdq[],
                               int dvLen,
                               TacsScalar dfdx[] );

  /**
     Get the number of design variables per node.

     The value defaults to one, unless over-ridden by the model
  */
  int getDesignVarsPerNode(){
    TACSElementModel *model = getElementModel();
    if (model){
      model->getDesignVarsPerNode();
    }
    // what are we doing with the 'model' object?
    return 2;
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
    if (dvNums){
      dvNums[0] = 0; // mass m
      dvNums[1] = 1; // stiffness k
    }
    // what to do with dvLen
    return 2;
  }

  /**
     Get the element design variables values

     @param elemIndex The local element index
     @param dvLen The length of the design array
     @param dvs The design variable values
     @return The number of design variable numbers defined by the element
  */
  int getDesignVars( int elemIndex,
                     int dvLen, TacsScalar dvs[] ){
    dvs[0] = this->m;
    dvs[1] = this->k;
    return 2;
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
    m = dvs[0];
    k = dvs[1];    
    return 2;
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
    // mass bounds
    lowerBound[0] = 1.0;
    upperBound[0] = 5.0;

    // stiffness bounds
    lowerBound[1] = 2.0;
    upperBound[1] = 10.0;

    return 2;
  }

  /**
     Add the derivative of the adjoint-residual product to the output vector

     This adds the contribution scaled by an input factor as follows:

     dvSens += scale*d(psi^{T}*(res))/dx

     By default the code is not implemented, but is not required so that
     analysis can be performed. Correct derivatives require a specific
     implementation.

     @param elemIndex The local element index
     @param time The simulation time
     @param scale The coefficient for the derivative result
     @param psi The element adjoint variables
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param ddvars The second time derivative of the element DOF
     @param dvLen The length of the design variable vector
     @param dvSens The derivative vector
  */
  void addAdjResProduct( int elemIndex, double time,
                         TacsScalar scale,
                         const TacsScalar psi[],
                         const TacsScalar Xpts[],
                         const TacsScalar vars[],
                         const TacsScalar dvars[],
                         const TacsScalar ddvars[],
                         int dvLen,
                         TacsScalar dfdx[] );

  int getVarsPerNode(){
    return 1;
  };
  
  int getNumNodes() {
    return 1;
  }

  double m, c, k;
};
