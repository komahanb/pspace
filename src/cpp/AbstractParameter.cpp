#include "AbstractParameter.h"

/**
   Constructor for creating probabilistic parameters
*/
AbstractParameter::AbstractParameter(){
  this->gauss = new GaussianQuadrature();
  this->polyn = new OrthogonalPolynomials();
}

/**
   Destructor for probabilistic parameters
*/
AbstractParameter::~AbstractParameter(){
  if(gauss){delete gauss;};
  if(polyn){delete polyn;};
}

/**
   Sets the ID for the created parameter
   @param pid parameter ID
*/
void AbstractParameter::setParameterID(int pid){
  this->parameter_id = pid;
}

/**
   Returns the ID for the created parameter
*/
int AbstractParameter::getParameterID(){
  return this->parameter_id;
}

/**
   Sets the maximum degree of expansion for created parameters

   @param dmax degree of expansion for parameter
*/
void AbstractParameter::setMaxDegree(int dmax){
  this->dmax = dmax;
}

/**
   Returns the maximum degree of created parameter
*/
int AbstractParameter::getMaxDegree(){
  return this->dmax;
}
