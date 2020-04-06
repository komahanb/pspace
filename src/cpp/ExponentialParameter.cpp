#include "ExponentialParameter.h"

/**
  Construct exponential parameter with input parameters

  @param pid parameter_id
  @param mu location of the parameter
  @param beta stretch of the parameter
*/
ExponentialParameter::ExponentialParameter(int pid, scalar mu, scalar beta)
  : AbstractParameter() {
  this->setParameterID(pid);
  this->mu = mu;
  this->beta = beta;
}

/**
  Destructor
*/
ExponentialParameter::~ExponentialParameter(){}

/**
   Returns the quadrature point and weights

  @param npoints number of quadrature points in 1-D quadrature
  @param z array of points in standard quadraure
  @param y array of points in general quadraure
  @param w array of weights for each point
*/
void ExponentialParameter::quadrature(int npoints, scalar *z, scalar *y, scalar *w){
  this->gauss->laguerreQuadrature(npoints,
                                 this->mu, this->beta,
                                 z, y, w);
}

/**
  Evaluate the basis of order d at the point as L(z,d)

  @param z point to evaluate the basis function
  @param d degree of basis function
*/
scalar ExponentialParameter::basis(scalar z, int d){
  return this->polyn->unit_laguerre(z, d);
}
