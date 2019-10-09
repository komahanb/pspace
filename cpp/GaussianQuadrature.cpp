/*
  Class to return points and weights for Gaussian Quadrature
  
  Author: Komahan Boopathy (komahanboopathy@gmail.com)
*/

#include <stdio.h>
#include <math.h>
#include "GaussianQuadrature.h"

//===================================================================//
// Constructor and Destructor
//===================================================================//

/*
  Constructor
*/
GaussianQuadrature::GaussianQuadrature(){}

/*
  Destructor
*/
GaussianQuadrature::~GaussianQuadrature(){}


//===================================================================//
// Public functions
//===================================================================//

/*
  Return hermite quadrature points and weights
*/
void GaussianQuadrature::hermiteQuadrature(int npoints, 
                                       double mu, double sigma, 
                                       double *z, double *y, double *w){

}

/*
  Return legendre quadrature points and weights
*/
void GaussianQuadrature::legendreQuadrature(int npoints, 
                                        double a, double b, 
                                        double *z, double *y, double *w){

}

/*
  Return laguerre quadrature points and weights
*/
void GaussianQuadrature::laguerreQuadrature(int npoints, 
                                        double mu, double beta, 
                                        double *z, double *y, double *w){

}

int main(int argc, char *argv[] ){

  // Create quadrature object
  GaussianQuadrature *gaussQuad = new GaussianQuadrature();
  
  // Allocate memory on heap
  int npoints = 10;
  double *z = new double[npoints];
  double *y = new double[npoints];
  double *w = new double[npoints];

  // Hermite Quadrature
  double mun = 0.0;
  double sigman = 1.0;
  gaussQuad->hermiteQuadrature(npoints, mun, sigman, 
                                &z[0], &y[0], &w[0]);  
  printf("Hermite Quadrature\n");
  for ( int i = 0 ; i < npoints; i++ ){
    printf("%d %15.6f %15.6f %15.6f\n", i, z[i], y[i], w[i]);
  }

  // Legendre Quadrature
  double a = 0.0;
  double b = 1.0;
  gaussQuad->legendreQuadrature(npoints, a, b, 
                                 &z[0], &y[0], &w[0]);  
  printf("\nLegendre Quadrature\n");
  for ( int i = 0 ; i < npoints; i++ ){
    printf("%d %15.6f %15.6f %15.6f\n", i, z[i], y[i], w[i]);
  }

  // Laguerre Quadrature
  double mu = 0.0;
  double beta = 1.0;
  gaussQuad->laguerreQuadrature(npoints, mu, beta, 
                                 &z[0], &y[0], &w[0]);  
  printf("\nLaguerre Quadrature\n");
  for ( int i = 0 ; i < npoints; i++ ){
    printf("%d %15.6f %15.6f %15.6f\n", i, z[i], y[i], w[i]);
  }

  // Free allocated heap memory
  delete gaussQuad;
  delete[] z;
  delete[] y;
  delete[] w;
}
