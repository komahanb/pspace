#include "TACSMutableElement3D.h"
#include "TACSLinearElasticity.h"

TACSMutableElement3D::TACSMutableElement3D( TACSElementModel *_model, TACSElementBasis *_basis ) 
  : TACSElement3D(_model, _basis) {}

TACSMutableElement3D::~TACSMutableElement3D() {}

void TACSMutableElement3D::setDensity( TacsScalar *_rho ){
  TACSLinearElasticity3D *model = dynamic_cast<TACSLinearElasticity3D*>(this->getElementModel());
  if (!model) return;
  TMROctConstitutive *con = dynamic_cast<TMROctConstitutive*>(model->getConstitutive());
  if (!con) return;
}


