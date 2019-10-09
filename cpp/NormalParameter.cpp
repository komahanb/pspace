#include "NormalParameter.h"

/*
  Construct normal parameter with input parameters
*/
NormalParameter::NormalParameter(int pid, double mu, double sigma){
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
void NormalParameter::quadrature(int npoints, double *z, double *y, double *w){
  this->gauss->hermiteQuadrature(npoints, 
                                 this->mu, this->sigma, 
                                 z, y, w);
}

/*
  Evalute the basis of order d at the point z
*/
void NormalParameter::basis(double z, int d){
  this->polyn->unit_hermite(z, d);
}
