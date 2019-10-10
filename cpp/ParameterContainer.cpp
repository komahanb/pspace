#include"ParameterContainer.h"

using namespace std;

ParameterContainer::ParameterContainer(){
  this->num_parameters = 0;   
}
ParameterContainer::~ParameterContainer(){}

/*
  Add the parameter into  collection
*/
void ParameterContainer::addParameter(AbstractParameter **param){
  if (this->num_parameters > 5){
    printf("Parameter container is full");
  }
  // this->pmap.insert(pair<int, AbstractParameter>(this->num_parameters,param));
  // this->num_parameters += 1;
}

int ParameterContainer::getNumBasisTerms(){
  return this->num_basis_terms;
}

int ParameterContainer::getNumParameters(){
  return this->num_parameters;
}

int ParameterContainer::getNumQuadraturePoints(){
  return this->num_quadrature_points;
}

void  ParameterContainer::initializeBasis(){}

void  ParameterContainer::initializeQuadrature(){}
