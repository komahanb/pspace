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
  AbstractParameter* createNormalParameter( double mu, double sigma,
                                            void (*setfunc)(void*, double),
                                            double (*getfunc)(void*) );
  AbstractParameter* createUniformParameter( double a, double b,
                                             void (*setfunc)(void*, double),
                                             double (*getfunc)(void*) );
  AbstractParameter* createExponentialParameter( double mu, double beta,
                                                 void (*setfunc)(void*, double),
                                                 double (*getfunc)(void*) );
  
 private:
  int next_parameter_id;
};

