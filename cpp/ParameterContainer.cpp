#include"ParameterContainer.h"
#include"ParameterFactory.h"

ParameterContainer::ParameterContainer(){
  this->num_parameters = 0;
}
ParameterContainer::~ParameterContainer(){}

/*
  Add the parameter into  collection
*/
void ParameterContainer::addParameter(AbstractParameter *param){
  if (this->num_parameters > 4){
    printf("Warning: Parameter container is full\n");
  }
  this->pmap.insert(pair<int, AbstractParameter*>(param->getParameterID(), param));
  this->num_parameters++;
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

int main( int argc, char *argv[] ){
  // Create random parameters
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *p1 = factory->createNormalParameter(-4.0, 0.5);
  AbstractParameter *p2 = factory->createUniformParameter(-5.0, 4.0);
  AbstractParameter *p3 = factory->createExponentialParameter(6.0, 1.0);
  AbstractParameter *p4 = factory->createExponentialParameter(6.0, 1.0);
  AbstractParameter *p5 = factory->createNormalParameter(-4.0, 0.5);
   
  // Create container and add random paramters
  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(p1);
  pc->addParameter(p2);
  pc->addParameter(p3);
  pc->addParameter(p4);
  pc->addParameter(p5);
  
  return 1;
}
