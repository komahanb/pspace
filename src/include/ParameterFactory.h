#include"AbstractParameter.h"
#include"NormalParameter.h"
#include"UniformParameter.h"
#include"ExponentialParameter.h"

#include "scalar.h"

/**
   Factory class for creating new probabilistically modeled random
   parameters
 */
class ParameterFactory {
 public:
  // Constructor and Destructor
  ParameterFactory();
  ~ParameterFactory();

  // Creates different parameter types
  AbstractParameter* createNormalParameter( scalar mu, scalar sigma );
  AbstractParameter* createUniformParameter( scalar a, scalar b );
  AbstractParameter* createExponentialParameter( scalar mu, scalar beta );

  // Overloaded constructors
  AbstractParameter* createNormalParameter(scalar mu, scalar sigma, int dmax);
  AbstractParameter* createUniformParameter(scalar a, scalar b, int dmax);
  AbstractParameter* createExponentialParameter(scalar mu, scalar beta, int dmax);

 private:
  int next_parameter_id;
};
