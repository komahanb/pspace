#include "NormalParameter.h"

/**
  Construct normal parameter with input parameters

  @param pid parameter ID
  @param mu mean of the parameter
  @param sigma standard deviation of the parameter
*/
NormalParameter::NormalParameter(int pid, scalar mu, scalar sigma)
  : AbstractParameter() {
  this->setParameterID(pid);
  this->mu = mu;
  this->sigma = sigma;
}

/**
  Destructor
*/
NormalParameter::~NormalParameter(){}

/**
  Returns the quadrature point and weights

  @param npoints number of quadrature points in 1-D quadrature
  @param z array of points in standard quadraure
  @param y array of points in general quadraure
  @param w array of weights for each point
*/
void NormalParameter::quadrature(int npoints, scalar *z, scalar *y, scalar *w){
  this->gauss->hermiteQuadrature(npoints,
                                 this->mu, this->sigma,
                                 z, y, w);
}

/**
  Evalute the basis of order d at the point z H(z,d)

  @param z point to evaluate the basis function
  @param d degree of basis function
*/
scalar NormalParameter::basis(scalar z, int d){
  return this->polyn->unit_hermite(z, d);
}
