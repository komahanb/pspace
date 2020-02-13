#include "smd.h"

void updateSMD( TACSElement *elem, TacsScalar *vals ){
  SMD *smd = dynamic_cast<SMD*>(elem);
  if (smd != NULL) {
    smd->c = vals[0];
  } else {
    printf("Element mismatch while updating...");
  }
}

SMD::SMD(TacsScalar m, TacsScalar c, TacsScalar k){
  this->m = m;
  this->c = c;
  this->k = k;  
}

SMD::~SMD(){
  printf("Decrefing SMD deterministic element\n");
}

void SMD::getInitConditions( int elemIndex, const TacsScalar X[],
                             TacsScalar v[], TacsScalar dv[], TacsScalar ddv[] ){
  int num_vars = getNumNodes()*getVarsPerNode();
  memset(v, 0, num_vars*sizeof(TacsScalar));
  memset(dv, 0, num_vars*sizeof(TacsScalar));
  memset(ddv, 0, num_vars*sizeof(TacsScalar));

  // set init conditions
  v[0] = 1.0;
  dv[0] = 0.0;
}

void SMD::addResidual( int elemIndex, double time,
                       const TacsScalar X[], const TacsScalar v[],
                       const TacsScalar dv[], const TacsScalar ddv[],
                       TacsScalar res[] ){
  res[0] += m*ddv[0] + c*dv[0] + k*v[0];
}

void SMD::addJacobian( int elemIndex, double time,
                       TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                       const TacsScalar X[], const TacsScalar v[],
                       const TacsScalar dv[], const TacsScalar ddv[],
                       TacsScalar res[], TacsScalar mat[] ){
  addResidual(elemIndex, time, X, v, dv, ddv, res);
  mat[0] += gamma*m + beta*c + alpha*k;
}

int SMD::evalPointQuantity( int elemIndex, int quantityType,
                            double time,
                            int n, double pt[],
                            const TacsScalar Xpts[],
                            const TacsScalar v[],
                            const TacsScalar dv[],
                            const TacsScalar ddv[],
                            TacsScalar *quantity ){
  if (quantityType == TACS_KINETIC_ENERGY_FUNCTION){
    *quantity = 0.5*m*dv[0]*dv[0];
    return 1;
  } else  if (quantityType == TACS_POTENTIAL_ENERGY_FUNCTION){
    *quantity = 0.5*k*v[0]*v[0];
    return 1;
  } else  if (quantityType == TACS_DISPLACEMENT_FUNCTION){
    *quantity = v[0];
    return 1;
  } else  if (quantityType == TACS_VELOCITY_FUNCTION){
    *quantity = dv[0];
    return 1;
  }
  return 0;
}

void SMD::addPointQuantitySVSens( int elemIndex, int quantityType,
                                  double time,
                                  TacsScalar alpha,
                                  TacsScalar beta,
                                  TacsScalar gamma,
                                  int n, double pt[],
                                  const TacsScalar Xpts[],
                                  const TacsScalar v[],
                                  const TacsScalar dv[],
                                  const TacsScalar ddv[],
                                  const TacsScalar dfdq[],
                                  TacsScalar dfdu[] ){
  if (quantityType == TACS_KINETIC_ENERGY_FUNCTION){
    dfdu[0] += beta*m*dv[0];
  } else  if (quantityType == TACS_POTENTIAL_ENERGY_FUNCTION){
    dfdu[0] += alpha*k*v[0];
  } else  if (quantityType == TACS_DISPLACEMENT_FUNCTION){
    dfdu[0] += alpha;
  } else  if (quantityType == TACS_VELOCITY_FUNCTION){
    dfdu[0] += beta;
  }
}

/*
  Implementation for dFdx
*/
void SMD::addPointQuantityDVSens( int elemIndex, int quantityType,
                                  double time,
                                  TacsScalar scale,
                                  int n, double pt[],
                                  const TacsScalar Xpts[],
                                  const TacsScalar v[],
                                  const TacsScalar dv[],
                                  const TacsScalar ddv[],
                                  const TacsScalar dfdq[],
                                  int dvLen,
                                  TacsScalar dfdx[] ){
  if (quantityType == TACS_KINETIC_ENERGY_FUNCTION){
    dfdx[0] += scale*0.5*dv[0]*dv[0];
    dfdx[1] += 0.0;
  } else  if (quantityType == TACS_POTENTIAL_ENERGY_FUNCTION){
    dfdx[0] += 0.0;
    dfdx[1] += scale*0.5*v[0]*v[0];
  } else  if (quantityType == TACS_DISPLACEMENT_FUNCTION){
    dfdx[0] += 0.0;
    dfdx[1] += 0.0;
  } else  if (quantityType == TACS_VELOCITY_FUNCTION){
    dfdx[0] += 0.0;
    dfdx[1] += 0.0;
  }
}

/*
  Adjoint residual product
*/
void SMD::addAdjResProduct( int elemIndex, double time,
                            TacsScalar scale,
                            const TacsScalar psi[],
                            const TacsScalar Xpts[],
                            const TacsScalar v[],
                            const TacsScalar dv[],
                            const TacsScalar ddv[],
                            int dvLen, 
                            TacsScalar dfdx[] ){
  dfdx[0] += scale*psi[0]*ddv[0];
  dfdx[1] += scale*psi[0]*v[0]; 
}
