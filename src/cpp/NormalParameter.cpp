#include "NormalParameter.h"

/*
  Construct normal parameter with input parameters
*/
NormalParameter::NormalParameter(int pid, scalar mu, scalar sigma)
  : AbstractParameter() {
  this->setParameterID(pid);
  this->mu = mu;
  this->sigma = sigma;
}

/*
  Destructor
*/
NormalParameter::~NormalParameter(){}

/*
  Evaluate the basis at the supplied point 
*/
void NormalParameter::quadrature(int npoints, scalar *z, scalar *y, scalar *w){
  this->gauss->hermiteQuadrature(npoints, 
                                 this->mu, this->sigma, 
                                 z, y, w);
}

/*
  Evalute the basis of order d at the point z
*/
scalar NormalParameter::basis(scalar z, int d){
  return this->polyn->unit_hermite(z, d);
}
