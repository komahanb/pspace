#include "TACSMutableElement3D.h"

// Constructor
TACSMutableElement3D::TACSMutableElement3D( TACSElement *_elem ){
  this->element = _elem;
  this->element->incref();
}

// Destructor
TACSMutableElement3D::~TACSMutableElement3D(){
  this->element->decref();
}

// Setter for density
void TACSMutableElement3D::setDensity( TacsScalar _rho ){
  TACSLinearElasticity3D *model = dynamic_cast<TACSLinearElasticity3D*>(this->getElementModel());
  if (!model) return;
  printf("model OK\n");
  TMROctConstitutive *con = dynamic_cast<TMROctConstitutive*>(model->getConstitutive());
  if (!con) return;
  printf("con OK\n");
  TMRStiffnessProperties *stiff = con->getStiffnessProperties();
  printf("stiff OK\n");
  TACSMaterialProperties **props = stiff->getMaterialProperties();
  printf("props OK\n");
  props[0]->setDensity(_rho);
  printf("set density OK %e \n", _rho);    
}

