#include "UniformParameter.h"

/*
  Construct uniform parameter with input parameters
*/
UniformParameter::UniformParameter(int pid, double a, double b)
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
void UniformParameter::quadrature(int npoints, double *z, double *y, double *w){
  this->gauss->legendreQuadrature(npoints, 
                                  this->a, this->b, 
                                  z, y, w);
}

/*
  Evalute the basis of order d at the point z
*/
void UniformParameter::basis(double z, int d){
  this->polyn->unit_legendre(z, d);
}
