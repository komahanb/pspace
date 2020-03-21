#include"ParameterFactory.h"

ParameterFactory::ParameterFactory(){
  this->next_parameter_id = 0;
}

ParameterFactory::~ParameterFactory(){}

AbstractParameter* ParameterFactory::createNormalParameter(scalar mu, scalar sigma)
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  return (AbstractParameter*) new NormalParameter(pid, mu, sigma);  
}

AbstractParameter* ParameterFactory::createUniformParameter(scalar a, scalar b)
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  return (AbstractParameter*) new UniformParameter(pid, a, b);  
}

AbstractParameter* ParameterFactory::createExponentialParameter(scalar mu, scalar beta)
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  return (AbstractParameter*) new ExponentialParameter(pid, mu, beta);  
}


AbstractParameter* ParameterFactory::createNormalParameter( scalar mu, scalar sigma,
                                                            int dmax)
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  AbstractParameter *param =  (AbstractParameter*) new NormalParameter(pid, mu, sigma); 
  param->setMaxDegree(dmax);
  return param;
}

AbstractParameter* ParameterFactory::createUniformParameter( scalar a, scalar b,
                                                             int dmax )
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  AbstractParameter *param = (AbstractParameter*) new UniformParameter(pid, a, b);  
  param->setMaxDegree(dmax);
  return param;
}

AbstractParameter* ParameterFactory::createExponentialParameter( scalar mu, scalar beta,
                                                                 int dmax )
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  AbstractParameter *param = (AbstractParameter*) new ExponentialParameter(pid, mu, beta);
  param->setMaxDegree(dmax);
  return param;
}
