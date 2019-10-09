#include "AbstractParameter.h"

/*
  Constructor
*/
AbstractParameter::AbstractParameter(){
  this->gauss = new GaussianQuadrature();
  this->polyn = new OrthogonalPolynomials();
}

/*
  Destructor
*/
AbstractParameter::~AbstractParameter(){
  if(gauss){delete gauss;};
  if(polyn){delete polyn;};
}

/*
  Setter for parameter ID
*/
void AbstractParameter::setParameterID(int pid){
  this->parameter_id = pid;
}

/*
  Getter for parameter ID
*/
int AbstractParameter::getParameterID(){
  return this->parameter_id;
}

