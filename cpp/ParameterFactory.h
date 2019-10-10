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
  AbstractParameter* createNormalParameter(double mu, double sigma);
  AbstractParameter* createUniformParameter(double a, double b);
  AbstractParameter* createExponentialParameter(double mu, double);
  
 private:
  int next_parameter_id;
};

