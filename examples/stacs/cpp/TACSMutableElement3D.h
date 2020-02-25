/*
  A TACSElement that allows its attributes to change for stochastic
  computations
*/

#ifndef TACS_MUTABLE_ELEMENT_H
#define TACS_MUTABLE_ELEMENT_H

#include "TACSElement.h"
#include "TACSElementModel.h"
#include "TACSElementBasis.h"
#include "TMROctConstitutive.h"

class TACSMutableElement3D : public TACSElement {
 public:
  TACSMutableElement3D( TACSElement *_elem );

  //  ~TACSMutableElement();
  /* TACSMutableElement3D( TACSElementModel *_model, TACSElementBasis *_basis ); */
  /* ~TACSMutableElement3D(); */
  /* TACSElement( int _componentNum = 0 ){ */
  /*   this->elem-> */
  /*   componentNum = _componentNum; */
  /* } */
  //virtual ~TACSElement(){}

  void setDensity( TacsScalar *_rho );

  /**
     Set the component number for this element.

     The component number can be used to identify groups of elements
     for visualization purposes

     @param comp_num The component number assigned to the element
  */
  void setComponentNum( int comp_num ){
    this->element->setComponentNum(comp_num);
  }

  /**
     Get the component number for this element

     @return The component number for the element
  */
  int getComponentNum(){
    return this->element->getComponentNum();
  }

  /**
     Get a string representation of the element name

     @return The name of the element
  */
  const char* getObjectName(){
    return this->element->getObjectName();
  }

  /**
     Get the number of degrees of freedom per node for this element

     @return The number of degrees of freedom per node
  */
  virtual int getVarsPerNode(){
    return this->element->getVarsPerNode();
  };

  /**
     Get the number of nodes associated with this element

     @return The number of nodes for this element
  */
  virtual int getNumNodes(){
    return this->element->getNumNodes();
  }

  /**
     Get the number of variables owned by the element
  */
  int getNumVariables(){
    return this->element->getNumVariables();
  }

  /**
     Get the node index where a Lagrange multiplier is defined.

     A negative index indicates that no multiplier is defined. The
     index is relative to the ordering in the element.

     @return Index of a Lagrange multiplier node
  */
  virtual int getMultiplierIndex(){
    return this->element->getMultiplierIndex();
  }

  /**
     Get the element basis class

     @return The TACSElementBasis class associated with this element. Possibly NULL.
  */
  virtual TACSElementBasis* getElementBasis(){
    return this->element->getElementBasis();
  }

  /**
     Get the element model class

     @return The TACSElementModel class associated with this element. Possibly NULL.
  */
  virtual TACSElementModel* getElementModel(){
    return this->element->getElementModel();
  }

  /**
     Get the type of element layout for visualization

     @return The layout type for this element
  */
  virtual ElementLayout getLayoutType(){
    return this->element->getLayoutType();
  }

  /**
     Get the number of design variables per node.

     The value defaults to one, unless over-ridden by the model
  */
  virtual int getDesignVarsPerNode(){
    return this->element->getDesignVarsPerNode();
  }

  /**
     Retrieve the global design variable numbers associated with this element

     Note when the dvNums argument is NULL, then the result is a query
     on the number of design variables and the array is not set.

     @param dvLen The length of the array dvNums
     @param dvNums An array of the design variable numbers for this element
     @return The number of design variable numbers defined by the element
  */
  virtual int getDesignVarNums( int elemIndex, int dvLen, int dvNums[] ){
    return this->element->getDesignVarNums(elemIndex, dvLen, dvNums);
  }

  /**
     Set the element design variables from the design vector

     @param elemIndex The local element index
     @param dvLen The length of the design array
     @param dvs The design variable values
     @return The number of design variable numbers defined by the element
  */
  virtual int setDesignVars( int elemIndex,
                             int dvLen, const TacsScalar dvs[] ){
    return this->element->setDesignVars(elemIndex, dvLen, dvs);
  }

  /**
     Get the element design variables values

     @param elemIndex The local element index
     @param dvLen The length of the design array
     @param dvs The design variable values
     @return The number of design variable numbers defined by the element
  */
  virtual int getDesignVars( int elemIndex,
                             int dvLen, TacsScalar dvs[] ){
    return this->element->getDesignVars(elemIndex, dvLen, dvs);
  }

  /**
     Get the lower and upper bounds for the design variable values

     @param elemIndex The local element index
     @param dvLen The length of the design array
     @param lowerBound The design variable lower bounds
     @param lowerBound The design variable upper bounds
     @return The number of design variable numbers defined by the element
  */
  virtual int getDesignVarRange( int elemIndex, int dvLen,
                                 TacsScalar lowerBound[],
                                 TacsScalar upperBound[] ){
    return this->element->getDesignVarRange(elemIndex, dvLen, lowerBound, upperBound);
  }

  /**
     Retrieve the initial conditions for time-dependent analysis

     By default, the initial displacements, velocities and accelerations
     are zero.

     @param elemIndex The local element index
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param ddvars The second time derivative of the element DOF
  */
  virtual void getInitConditions( int elemIndex,
                                  const TacsScalar Xpts[],
                                  TacsScalar vars[],
                                  TacsScalar dvars[],
                                  TacsScalar ddvars[] ){
    this->element->getInitConditions(elemIndex, Xpts, vars, dvars, ddvars);
  }

  /**
     Add the contributions to the derivative from the initial conditions

     @param elemIndex The local element index
     @param Xpts The element node locations
     @param adjVars The values of the element adjoint
     @param adjDVars The adjoint of the first time derivatives
     @param adjDDVars The adjoint of the first time derivatives
     @param ddvars The second time derivative of the element DOF
     @param dvLen The length of the design vector
  */
  virtual void addInitConditionAdjResProduct( int elemIndex,
                                              const TacsScalar Xpts[],
                                              const TacsScalar adjVars[],
                                              const TacsScalar adjDVars[],
                                              const TacsScalar adjDDVars[],
                                              int dvLen,
                                              TacsScalar fdvSens[] ){
    this->element->addInitConditionAdjResProduct(elemIndex, Xpts, adjVars, adjDVars, adjDDVars, dvLen, fdvSens);
  }

  /**
     Get the contribution to the derivatives of the initial conditions w.r.t.
     the node locations

     @param elemIndex The local element index
     @param Xpts The element node locations
     @param adjVars The values of the element adjoint
     @param adjDVars The adjoint of the first time derivatives
     @param adjDDVars The adjoint of the first time derivatives
     @param fXptSens Derivative w.r.t. the node locations
  */
  virtual void getInitConditionAdjResXptProduct( int elemIndex,
                                                 const TacsScalar Xpts[],
                                                 const TacsScalar adjVars[],
                                                 const TacsScalar adjDVars[],
                                                 const TacsScalar adjDDVars[],
                                                 TacsScalar fXptSens[] ){
    this->element->getInitConditionAdjResXptProduct(elemIndex, Xpts, adjVars, adjDVars, adjDDVars, fXptSens);
  }

  /**
     Compute the kinetic and potential energy within the element.

     This can be used to evaluate the Hamiltonian and test whether the
     element satisfies the Lagrangian equations of motion.

     @param elemIndex The local element index
     @param time The simulation time
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param Te The kinetic energy contributed by this element
     @param Pe the potential energy contributed by this element
  */
  virtual void computeEnergies( int elemIndex,
                                double time,
                                const TacsScalar Xpts[],
                                const TacsScalar vars[],
                                const TacsScalar dvars[],
                                TacsScalar *Te,
                                TacsScalar *Pe ){
    this->element->computeEnergies(elemIndex, time, Xpts, vars, dvars, Te, Pe);
  }

  /**
     Add the contribution from this element to the residual.

     Note that this simply adds, and does not over-write the residual so
     that multiple contributions can be computed.

     @param elemIndex The local element index
     @param time The simulation time
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param ddvars The second time derivative of the element DOF
     @param res The element residual input/output
  */
  virtual void addResidual( int elemIndex, double time,
                            const TacsScalar Xpts[],
                            const TacsScalar vars[],
                            const TacsScalar dvars[],
                            const TacsScalar ddvars[],
                            TacsScalar res[] ) {
    this->element->addResidual(elemIndex, time, Xpts, vars, dvars, ddvars, res);
  }

  /**
     Add the contribution from this element to the residual and Jacobian.

     Note that this simply adds, and does not over-write the Jacobian so
     that multiple contributions can be computed.

     The Jacobian contribution consists of a linear combination of the
     Jacobians with respect to the variables, and their first and second
     time derivatives as follows:

     mat += alpha*d(res)/d(vars) + beta*d(res)/d(dvars) + gamma*d(res)/d(ddvars)

     @param elemIndex The local element index
     @param time The simulation time
     @param alpha The coefficient for the DOF Jacobian
     @param beta The coefficient for the first time derivative DOF Jacobian
     @param gamma The coefficient for the second time derivative DOF Jacobian
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param ddvars The second time derivative of the element DOF
     @param res The element residual input/output
     @param mat The element Jacobian input/output
  */
  virtual void addJacobian( int elemIndex, double time,
                            TacsScalar alpha,
                            TacsScalar beta,
                            TacsScalar gamma,
                            const TacsScalar Xpts[],
                            const TacsScalar vars[],
                            const TacsScalar dvars[],
                            const TacsScalar ddvars[],
                            TacsScalar res[],
                            TacsScalar mat[] ){
    this->element->addJacobian(elemIndex, time, alpha, beta, gamma,
                               Xpts, vars, dvars, ddvars, res, mat);
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
  virtual void addAdjResProduct( int elemIndex, double time,
                                 TacsScalar scale,
                                 const TacsScalar psi[],
                                 const TacsScalar Xpts[],
                                 const TacsScalar vars[],
                                 const TacsScalar dvars[],
                                 const TacsScalar ddvars[],
                                 int dvLen,
                                 TacsScalar dfdx[] ){
    this->element->addAdjResProduct(elemIndex, time, scale,
                                    psi, Xpts, vars, dvars, ddvars, dvLen, dfdx);
  }

  /**
     Add the derivative of the adjoint-residual product to the output vector

     This adds the contribution scaled by an input factor as follows:

     dvSens += scale*d(psi^{T}*(res))/d(Xpts)

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
  virtual void addAdjResXptProduct( int elemIndex, double time,
                                    TacsScalar scale,
                                    const TacsScalar psi[],
                                    const TacsScalar Xpts[],
                                    const TacsScalar vars[],
                                    const TacsScalar dvars[],
                                    const TacsScalar ddvars[],
                                    TacsScalar fXptSens[] ){
    this->element->addAdjResXptProduct(elemIndex, time, scale, psi, Xpts,
                                       vars, dvars, ddvars, fXptSens);
  }

  /**
     Compute a specific type of element matrix (mass, stiffness, geometric
     stiffness, etc.)

     @param elemIndex The local element index
     @param matType The type of element matrix to compute
     @param Xpts The element node locations
     @param vars The values of element degrees of freedom
     @param mat The element matrix output
  */
  virtual void getMatType( int elemIndex, ElementMatrixType matType,
                           const TacsScalar Xpts[],
                           const TacsScalar vars[],
                           TacsScalar mat[] ){
    this->element->getMatType(elemIndex, matType, Xpts, vars, mat);
  }

  /**
     Add the derivative of the product of a specific matrix w.r.t.
     the design variables

     dvSens += scale*d(psi^{T}*(mat)*phi)/d(x)

     where mat is computed via the getMatType().

     @param elemIndex The local element index
     @param matType The type of element matrix to compute
     @param scale The scalar value that multiplies the derivative
     @param psi The left-hand vector
     @param phi The right-hand vector
     @param Xpts The element node locations
     @param vars The values of element degrees of freedom
     @param mat The element matrix output
  */
  virtual void addMatDVSensInnerProduct( int elemIndex,
                                         ElementMatrixType matType,
                                         TacsScalar scale,
                                         const TacsScalar psi[],
                                         const TacsScalar phi[],
                                         const TacsScalar Xpts[],
                                         const TacsScalar vars[],
                                         int dvLen,
                                         TacsScalar dvSens[] ){
    this->element->addMatDVSensInnerProduct(elemIndex, matType, scale,
                                            psi, phi, Xpts, vars, dvLen, dvSens);
  }

  /**
     Compute the derivative of the product of a specific matrix w.r.t.
     the input variables (vars).

     dvSens = d(psi^{T}*(mat)*phi)/d(vars)

     where mat is computed via the getMatType().
    
     @param elemIndex The local element index
     @param matType The type of element matrix to compute
     @param psi The left-hand vector
     @param phi The right-hand vector
     @param Xpts The element node locations
     @param vars The values of element degrees of freedom
     @param res The residual output The element matrix output
  */
  virtual void getMatSVSensInnerProduct( int elemIndex,
                                         ElementMatrixType matType,
                                         const TacsScalar psi[],
                                         const TacsScalar phi[],
                                         const TacsScalar Xpts[],
                                         const TacsScalar vars[],
                                         TacsScalar res[] ){
    this->element->getMatSVSensInnerProduct(elemIndex, matType,
                                            psi, phi, Xpts, vars, res);
  }

  /**
     Evaluate a point-wise quantity of interest.

     @param elemIndex The index of the element
     @param quantityType The integer indicating the pointwise quantity
     @param time The simulation time
     @param n The quadrature point index
     @param pt The quadrature point
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param ddvars The second time derivative of the element DOF
     @param quantity The output quantity of interest
     @return Integer indicating the number of defined quantities
  */
  virtual int evalPointQuantity( int elemIndex, int quantityType,
                                 double time,
                                 int n, double pt[],
                                 const TacsScalar Xpts[],
                                 const TacsScalar vars[],
                                 const TacsScalar dvars[],
                                 const TacsScalar ddvars[],
                                 TacsScalar *quantity ){
    return this->element->evalPointQuantity(elemIndex, quantityType, time,
                                            n, pt, Xpts, vars, dvars, ddvars, quantity);
  }

  /**
     Add the derivative of the point quantity w.r.t. the design variables

     @param elemIndex The index of the element
     @param quantityType The integer indicating the pointwise quantity
     @param time The simulation time
     @param n The quadrature point index
     @param pt The quadrature point
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param ddvars The second time derivative of the element DOF
     @param dvLen The length of the design array
     @param fdvSens The derivative array
  */
  virtual void addPointQuantityDVSens( int elemIndex, int quantityType,
                                       double time,
                                       TacsScalar scale,
                                       int n, double pt[],
                                       const TacsScalar Xpts[],
                                       const TacsScalar vars[],
                                       const TacsScalar dvars[],
                                       const TacsScalar ddvars[],
                                       const TacsScalar dfdq[],
                                       int dvLen,
                                       TacsScalar dfdx[] ){
    this->element->addPointQuantityDVSens(elemIndex, quantiyType, time, scale,
                                          n, pt,
                                          Xpts, vars, dvars, ddvars,
                                          dfdq, dvLen, dfdx);
  }

  /**
     Add the derivative of the point quantity w.r.t. the state variables

     @param elemIndex The index of the element
     @param time The simulation time
     @param quantityType The integer indicating the pointwise quantity
     @param alpha The coefficient for the state variables
     @param beta The coefficient for the first time derivatives
     @param gamma The coefficient for the second time derivatives
     @param n The quadrature point index
     @param pt The quadrature point
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param ddvars The second time derivative of the element DOF
     @param dvLen The length of the design array
     @param dfdu The derivative of the quantity w.r.t. state variables
  */
  virtual void addPointQuantitySVSens( int elemIndex, int quantityType,
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
                                       TacsScalar dfdu[] ){
    this->element->addPointQuantitySVSens(elemIndex, quantityType,
                                          time, alpha, beta, gamma,
                                          n, pt,
                                          Xpts, vars, dvars, ddvars, dfdq, dfdu);
  }

  /**
     Add the derivative of the point quantity w.r.t. the node locations

     @param elemIndex The index of the element
     @param quantityType The integer indicating the pointwise quantity
     @param time The simulation time
     @param scale The scalar factor applied to the derivative
     @param n The quadrature point index
     @param pt The quadrature point
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param ddvars The second time derivative of the element DOF
     @param dvLen The length of the design array
     @param dfdu The derivative of the quantity w.r.t. state variables
  */
  virtual void addPointQuantityXptSens( int elemIndex, int quantityType,
                                        double time,
                                        TacsScalar scale,
                                        int n, double pt[],
                                        const TacsScalar Xpts[],
                                        const TacsScalar vars[],
                                        const TacsScalar dvars[],
                                        const TacsScalar ddvars[],
                                        const TacsScalar dfdq[],
                                        TacsScalar dfdXpts[] ){
    this->element->addPointQuantityXptSens(elemIndex, quantityType, time, scale, n, pt,
                                           Xpts, vars, dvars, ddvars,
                                           dfdq, dfdXpts);
  }

  /**
     Compute the output data for visualization

     @param elemIndex The local element index
     @param etype The type of element data to be output
     @param write_flag The type of data to be output
     @param Xpts The element node locations
     @param vars The values of the element degrees of freedom
     @param dvars The first time derivative of the element DOF
     @param ddvars The second time derivative of the element DOF
     @param ld_data The dimension of the data
     @param data The data to be created
  */
  virtual void getOutputData( int elemIndex,
                              ElementType etype, int write_flag,
                              const TacsScalar Xpts[],
                              const TacsScalar vars[],
                              const TacsScalar dvars[],
                              const TacsScalar ddvars[],
                              int ld_data, TacsScalar *data ){
    this->element->getOutputData( elemIndex, etype,  write_flag,
                                  Xpts, vars, dvars, ddvars,
                                  ld_data, data );
  }

 private:
  int componentNum;
  TACSElement *element;
};

#endif // TACS_MUTABLE_ELEMENT_H
