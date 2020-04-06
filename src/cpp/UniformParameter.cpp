#include "UniformParameter.h"

/**
  Construct uniform parameter with input parameters

  @param pid parameter ID
  @param a lower bound
  @param b upper bound
*/
UniformParameter::UniformParameter(int pid, scalar a, scalar b)
  : AbstractParameter() {
  this->setParameterID(pid);
  this->a = a;
  this->b = b;
}

/**
  Destructor
*/
UniformParameter::~UniformParameter(){}

/**
  Returns the quadrature point and weights

  @param npoints number of quadrature points in 1-D quadrature
  @param z array of points in standard quadraure
  @param y array of points in general quadraure
  @param w array of weights for each point
*/
void UniformParameter::quadrature(int npoints, scalar *z, scalar *y, scalar *w){
  this->gauss->legendreQuadrature(npoints,
                                  this->a, this->b,
                                  z, y, w);
}

/**
  Evalute the basis of order d at the point z as P(z,d)

  @param z point to evaluate the basis function
  @param d degree of basis function
*/
scalar UniformParameter::basis(scalar z, int d){
  return this->polyn->unit_legendre(z, d);
}
