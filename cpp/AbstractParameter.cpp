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


void AbstractParameter::setMaxDegree(int dmax){
  this->dmax = dmax;
}
int AbstractParameter::getMaxDegree(){
  return this->dmax;
}
  
// Function pointer to set values
void AbstractParameter::setClientFunction( void (*func)(void*, double) ){
  this->set = func;
}

// Function pointer to retrieve values
void AbstractParameter::getClientFunction( double (*func)(void*) ){
  this->get = func;
}
  
void AbstractParameter::setClient(void *client){ 
  this->client = client;
}

void AbstractParameter::updateValue(void *obj, double value){
  printf("calling cpp update \n");
  if ( this->client == obj ){
    this->set(obj, value);
  } else {
    printf("skipping update \n");
  }
};
  
double AbstractParameter::getValue(void *obj){ 
  printf("calling cpp get \n");
  if ( this->client == obj ){
    return this->get(obj); 
  } else {
    printf("default return \n");
    return 0.0;
  }
}

