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
  if (!model){
    printf(">> Casting to TACSLinearElasticity3D failed\n");
    return;
  }
  TMROctConstitutive *con = dynamic_cast<TMROctConstitutive*>(model->getConstitutive());
  if (!con){
    printf(">> Casting to TMROctConstitutive failed\n");
    return;
  }
  TMRStiffnessProperties *stiff = con->getStiffnessProperties();
  TACSMaterialProperties **props = stiff->getMaterialProperties();
  props[0]->setDensity(_rho);
}

