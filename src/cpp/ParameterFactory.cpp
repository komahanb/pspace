#include"ParameterFactory.h"

/**
   Constructor for creating parameter factory
*/
ParameterFactory::ParameterFactory(){
  this->next_parameter_id = 0;
}

/**
   Destructor for parameter factory
*/
ParameterFactory::~ParameterFactory(){}

/**
   Create the normal parameter object and assign the next available ID
   for the parameter.

   @param mu mean
   @param sigma standard deviation
*/
AbstractParameter* ParameterFactory::createNormalParameter( scalar mu,
                                                            scalar sigma ){
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  return (AbstractParameter*) new NormalParameter(pid, mu, sigma);
}

/**
   Create the uniform parameter object and assign the next available ID
   for the parameter.

   @param a lower bound
   @param b upper bound
*/
AbstractParameter* ParameterFactory::createUniformParameter( scalar a,
                                                             scalar b ){
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  return (AbstractParameter*) new UniformParameter(pid, a, b);
}

/**
   Create the exponential parameter object and assign the next
   available ID for the parameter.

   @param mu location of parameter
   @param beta stretch of parameter
*/
AbstractParameter* ParameterFactory::createExponentialParameter( scalar mu,
                                                                 scalar beta ){
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  return (AbstractParameter*) new ExponentialParameter(pid, mu, beta);
}

/**
   Create the normal parameter object and assign the next available ID
   for the parameter.

   @param mu mean
   @param sigma standard deviation
   @param dmax degree of the parameter
*/
AbstractParameter* ParameterFactory::createNormalParameter( scalar mu,
                                                            scalar sigma,
                                                            int dmax){
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  AbstractParameter *param =  (AbstractParameter*) new NormalParameter(pid, mu, sigma);
  param->setMaxDegree(dmax);
  return param;
}

/**
   Create the uniform parameter object and assign the next available ID
   for the parameter.

   @param a lower bound
   @param b upper bound
   @param dmax degree of the parameter
*/
AbstractParameter* ParameterFactory::createUniformParameter( scalar a,
                                                             scalar b,
                                                             int dmax ){
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  AbstractParameter *param = (AbstractParameter*) new UniformParameter(pid, a, b);
  param->setMaxDegree(dmax);
  return param;
}

/**
   Create the exponential parameter object and assign the next
   available ID for the parameter.

   @param mu location of parameter
   @param beta stretch of parameter
   @param dmax degree of the parameter
*/
AbstractParameter* ParameterFactory::createExponentialParameter( scalar mu,
                                                                 scalar beta,
                                                                 int dmax ){
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  AbstractParameter *param = (AbstractParameter*) new ExponentialParameter(pid, mu, beta);
  param->setMaxDegree(dmax);
  return param;
}
