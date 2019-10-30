#include "AbstractParameter.h"

AbstractParameter::AbstractParameter(){
  this->gauss = new GaussianQuadrature();
  this->polyn = new OrthogonalPolynomials();
}
AbstractParameter::~AbstractParameter(){
  if(gauss){delete gauss;};
  if(polyn){delete polyn;};
}
void AbstractParameter::setParameterID(int pid){
  this->parameter_id = pid;
}
int AbstractParameter::getParameterID(){
  return this->parameter_id;
}
void AbstractParameter::setMaxDegree(int dmax){
  this->dmax = dmax;
}
int AbstractParameter::getMaxDegree(){
  return this->dmax;
}

