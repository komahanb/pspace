#include "UniformParameter.h"

/*
  Construct uniform parameter with input parameters
*/
UniformParameter::UniformParameter(int pid, scalar a, scalar b)
  : AbstractParameter() {
  this->setParameterID(pid);
  this->a = a;
  this->b = b;
}

/*
  Destructor
*/
UniformParameter::~UniformParameter(){}

/*
  Evaluate the basis at the supplied point 
*/
void UniformParameter::quadrature(int npoints, scalar *z, scalar *y, scalar *w){
  this->gauss->legendreQuadrature(npoints, 
                                  this->a, this->b, 
                                  z, y, w);
}

/*
  Evalute the basis of order d at the point z
*/
scalar UniformParameter::basis(scalar z, int d){
  return this->polyn->unit_legendre(z, d);
}
