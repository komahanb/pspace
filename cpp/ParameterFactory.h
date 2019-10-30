#include"AbstractParameter.h"
#include"NormalParameter.h"
#include"UniformParameter.h"
#include"ExponentialParameter.h"

class ParameterFactory {
 public:
  // Constructor and Destructor
  ParameterFactory();
  ~ParameterFactory();
  
  // Creates different parameter types
  AbstractParameter* createNormalParameter( double mu, double sigma );
  AbstractParameter* createUniformParameter( double a, double b );
  AbstractParameter* createExponentialParameter( double mu, double beta );

  // Overloaded constructors
  AbstractParameter* createNormalParameter( double mu, double sigma, int dmax);
  AbstractParameter* createUniformParameter( double a, double b, int dmax);
  AbstractParameter* createExponentialParameter( double mu, double beta, int dmax);
  
 private:
  int next_parameter_id;
};

