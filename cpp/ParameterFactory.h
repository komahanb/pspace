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
  void createNormalParameter(double mu, double sigma);
  void createUniformParameter(double a, double b);
  void createExponentialParameter(double mu, double);
  
 private:
  int next_parameter_id;
};

