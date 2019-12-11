#ifndef TACS_KSSTOCHASTIC_FUNCTION
#define TACS_KSSTOCHASTIC_FUNCTION

#include "TACSFunction.h"
#include "ParameterContainer.h"

// Define some quantities of interest
static const int FUNCTION_MEAN   = 0;
static const int FUNCTION_VARIANCE = 1;

class TACSKSStochasticFunction : public TACSFunction {
 public:
  TACSKSStochasticFunction( TACSAssembler *tacs,
                            TACSFunction *dfunc,
                            ParameterContainer *pc,
                            int quantityType,
                            int moment_type,
                            double ksWeight );
  ~TACSKSStochasticFunction();
  /**
     Get the object name
  */
  const char *getObjectName(){
    return this->dfunc->getObjectName();
  }
  
  /**
     Get the type of integration domain

     @return The enum type of domain
  */
  DomainType getDomainType(){
    return this->dfunc->getDomainType();
  }

  /**
     Get the stage type of this function: Either one or two stage

     Some functions (such as aggregation functionals) require a
     two-stage integration strategy for numerical stability.

     @return The enum type indicating whether this is a one or two stage func.
  */
  StageType getStageType(){
    return this->dfunc->getStageType();
  }
  
  /**
     Retrieve the element domain from the function

     @param elemNums The element numbers defining the domain
     @return The numer of elements in the domain
  */
  int getElementNums( const int **_elemNums ){
    this->dfunc->getElementNums(_elemNums);
  }
 
  /**
     Return the TACSAssembler object associated with this function
  */
  TACSAssembler *getAssembler(){
    return this->dfunc->getAssembler();
  }

  /**
     Initialize the function for the given type of evaluation

     This call is collective on all processors in the assembler.
  */
  void initEvaluation( EvaluationType ftype );

  /**
     Perform an element-wise integration over this element.

     Note that this is not a collective call and should be called once
     for each element within the integration domain.

     @param ftype The type of evaluation
     @param elemIndex The local element index
     @param element The TACSElement object
     @param time The simulation time
     @param scale The scalar integration factor to apply
     @param Xpts The element node locations
     @param vars The element DOF
     @param dvars The first time derivatives of the element DOF
     @param ddvars The second time derivatives of the element DOF
  */
  void elementWiseEval( EvaluationType ftype,
                        int elemIndex, TACSElement *element,
                        double time,
                        TacsScalar scale,
                        const TacsScalar Xpts[],
                        const TacsScalar vars[],
                        const TacsScalar dvars[],
                        const TacsScalar ddvars[] ); 
  /**
     Finalize the function evaluation for the specified eval type.
     
     This call is collective on all processors in the assembler.
  */
  void finalEvaluation( EvaluationType ftype );
  
  
  /**
     Get the value of the function
  */
  TacsScalar getFunctionValue();

  /**
     Evaluate the derivative of the function w.r.t. state variables
     
     @param elemIndex The local element index
     @param element The TACSElement object
     @param time The simulation time
     @param alpha Coefficient for the DOF derivative
     @param beta Coefficient for the first time DOF derivative
     @param gamma Coefficient for the second time DOF derivative
     @param Xpts The element node locations
     @param vars The element DOF
     @param dvars The first time derivatives of the element DOF
     @param ddvars The second time derivatives of the element DOF
  */
  void getElementSVSens( int elemIndex, TACSElement *element,
                         double time,
                         TacsScalar alpha, TacsScalar beta,
                         TacsScalar gamma,
                         const TacsScalar Xpts[],
                         const TacsScalar vars[],
                         const TacsScalar dvars[],
                         const TacsScalar ddvars[],
                         TacsScalar dfdu[] );

  /**
     Add the derivative of the function w.r.t. the design variables

     The design variables *must* be the same set of variables defined
     in the element. The TACSFunction class cannot define new design
     variables!

     @param elemIndex The local element index
     @param element The TACSElement object
     @param time The simulation time
     @param Xpts The element node locations
     @param vars The element DOF
     @param dvars The first time derivatives of the element DOF
     @param ddvars The second time derivatives of the element DOF
  */
  void addElementDVSens( int elemIndex, TACSElement *element,
                         double time, TacsScalar scale,
                         const TacsScalar Xpts[],
                         const TacsScalar vars[],
                         const TacsScalar dvars[],
                         const TacsScalar ddvars[],
                         int dvLen,
                         TacsScalar dfdx[] );
  /**
     Evaluate the derivative of the function w.r.t. the node locations

     @param elemIndex The local element index
     @param element The TACSElement object
     @param time The simulation time
     @param scale The scalar integration factor to apply
     @param Xpts The element node locations
     @param vars The element DOF
     @param dvars The first time derivatives of the element DOF
     @param ddvars The second time derivatives of the element DOF
  */
  virtual void getElementXptSens( int elemIndex, TACSElement *element,
                                  double time, TacsScalar scale,
                                  const TacsScalar Xpts[],
                                  const TacsScalar vars[],
                                  const TacsScalar dvars[],
                                  const TacsScalar ddvars[],
                                  TacsScalar dfdXpts[] ){
    int numNodes = element->getNumNodes();
    memset(dfdXpts, 0, 3*numNodes*sizeof(TacsScalar));
  }

 protected:

  TACSFunction *dfunc;
  ParameterContainer *pc;

 private:

  // The name of the function
  static const char *funcName;

  // Integer identifying the function type
  int quantityType;

  // The value of the KS weight
  double ksWeight;

  // Intermediate values in the functional evaluation
  TacsScalar *ksSum;
  TacsScalar *maxValue;
  TacsScalar *fvals;

  int moment_type;

  // number of stochastic quadrature points
  int nsqpts;
  int nsterms;

  MPI_Comm tacs_comm;
};

#endif
