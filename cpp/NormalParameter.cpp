#include "NormalParameter.h"
#include "GaussianQuadrature.h"

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
  
}

/*
  Evalute the basis of order d at the point z
*/
void NormalParameter::basis(double *z, int d){
  
}
