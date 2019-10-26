#include"ParameterFactory.h"

ParameterFactory::ParameterFactory(){}
ParameterFactory::~ParameterFactory(){}

AbstractParameter* ParameterFactory::createNormalParameter(double mu, double sigma)
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  return (AbstractParameter*) new NormalParameter(pid, mu, sigma);  
}

AbstractParameter* ParameterFactory::createUniformParameter(double a, double b)
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  return (AbstractParameter*) new UniformParameter(pid, a, b);  
}

AbstractParameter* ParameterFactory::createExponentialParameter(double mu, double beta)
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  return (AbstractParameter*) new ExponentialParameter(pid, mu, beta);  
}


AbstractParameter* ParameterFactory::createNormalParameter( double mu, double sigma,
                                                            void (*setfunc)(void*, double),
                                                            double (*getfunc)(void*) )
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  AbstractParameter *param =  (AbstractParameter*) new NormalParameter(pid, mu, sigma); 
  param->setClientFunction(setfunc);
  param->getClientFunction(getfunc);
  return param;
}

AbstractParameter* ParameterFactory::createUniformParameter( double a, double b,
                                                             void (*setfunc)(void*, double),
                                                             double (*getfunc)(void*) )
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  AbstractParameter *param = (AbstractParameter*) new UniformParameter(pid, a, b);  
  param->setClientFunction(setfunc);
  param->getClientFunction(getfunc);
  return param;
}

AbstractParameter* ParameterFactory::createExponentialParameter( double mu, double beta,
                                                                 void (*setfunc)(void*, double),
                                                                 double (*getfunc)(void*) )
{
  int pid = this->next_parameter_id;
  this->next_parameter_id++;
  AbstractParameter *param = (AbstractParameter*) new ExponentialParameter(pid, mu, beta);
  param->setClientFunction(setfunc);
  param->getClientFunction(getfunc);
  return param;
}


