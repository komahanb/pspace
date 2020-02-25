/*
  A TACSElement that allows its attributes to change for stochastic
  computations
*/

#ifndef TACS_MUTABLE_ELEMENT_3D_H
#define TACS_MUTABLE_ELEMENT_3D_H

#include "TACSElement3D.h"
#include "TACSElementModel.h"
#include "TACSElementBasis.h"
#include "TMROctConstitutive.h"

class TACSMutableElement3D : public TACSElement3D {
 public:
  TACSMutableElement3D( TACSElementModel *_model, TACSElementBasis *_basis );
  ~TACSMutableElement3D();
  void setDensity( TacsScalar *_rho );
};

#endif // TACS_MUTABLE_ELEMENT_3D_H
