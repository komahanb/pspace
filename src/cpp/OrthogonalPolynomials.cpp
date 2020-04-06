/*
  Class to evaluate orthogonal polynomials.

  Author: Komahan Boopathy (komahanboopathy@gmail.com)
*/

#include <stdio.h>
#include <math.h>

#include "OrthogonalPolynomials.h"

/**
   Constructor for orthogonal polynomials
*/
OrthogonalPolynomials::OrthogonalPolynomials(){}

/**
   Destructor for orthogonal polynomials
*/
OrthogonalPolynomials::~OrthogonalPolynomials(){}

/**
   Evaluate the Hermite polynomials

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::hermite(scalar z, int d){
  if ( d <= 4 ) {
    return explicit_hermite(z,d);
  } else {
    return recursive_hermite(z,d);
  }
}

/**
   Evaluate the unit Hermite polynomials

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::unit_hermite(scalar z, int d){
  return hermite(z,d)/sqrt(factorial(d));
}

/**
   Evaluate the Laguerre polynomials

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::laguerre(scalar z, int d){
  if ( d <= 4 ) {
    return explicit_laguerre(z,d);
  } else {
    return recursive_laguerre(z,d);
  }
}

/**
   Evaluate unit Laguerre polynomials (already normalized)

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::unit_laguerre(scalar z, int d){
  return laguerre(z,d);
}

/**
   Evaluate Legendre polynomials

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::legendre(scalar z, int d){
  if ( d <= 4 ) {
    return explicit_legendre(z,d);
  } else {
    return general_legendre(z,d);
  }
}

/**
   Evaluate unit legendre polynomials

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::unit_legendre(scalar z, int d){
  return legendre(z,d)*sqrt(scalar(2*d+1));
}

/**
   Combination nCr = n!/((n-r)!r!)
*/
scalar OrthogonalPolynomials::comb(int n, int r){
  scalar nfact  = factorial(n);
  scalar rfact  = factorial(r);
  scalar nrfact = factorial(n-r);
  return nfact/(rfact*nrfact);
}

/**
   Compute the factorial of a number

   @param n the numbr for which factorial is needed
*/
scalar OrthogonalPolynomials::factorial( int n ){
  scalar factorial;
  if ( n == 0 ){
    factorial = 1.0;
  } else if ( n == 1 ){
    factorial = 1.0;
  } else if ( n == 2 ){
    factorial = 2.0;
  } else if ( n == 3 ){
    factorial = 6.0;
  } else if ( n == 4 ){
    factorial = 24.0;
  } else if ( n == 5 ){
    factorial = 120.0;
  } else if ( n == 6 ){
    factorial = 720.0;
  } else if ( n == 7 ){
    factorial = 5040.0;
  } else if ( n == 8 ){
    factorial = 40320.0;
  } else if ( n == 9 ){
    factorial = 362880.0;
  } else if ( n == 10 ){
    factorial = 3628800.0;
  } else {
    factorial = 1.0;
    for ( int i = 1; i <= n; i++ ) {
      factorial *= scalar(i);
    }
  }
  return factorial;
}

/**
   Hermite polynomials are evaluated using explicit expressions

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::explicit_hermite(scalar z, int d){
  scalar hval = 0.0;
  if ( d == 0 ){
    hval = 1.0;
  } else if ( d == 1 ){
    hval = z;
  } else if ( d == 2 ){
    hval = z*z - 1.0;
  } else if ( d == 3 ){
    hval = z*z*z -3.0*z;
  } else if ( d == 4 ){
    hval = z*z*z*z -6.0*z*z + 3.0;
  }
  return hval;
}

/**
   Hermite polynomials are evaluated using recursive expressions

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::recursive_hermite(scalar z, int d){
  scalar hval = 0.0;
  if ( d == 0 ) {
    hval = 1.0;
  } else if ( d == 1 ) {
    hval = z;
  } else {
    hval = z*recursive_hermite(z,d-1) - scalar(d-1)*recursive_hermite(z,d-2);
  }
  return hval;
}

/**
   Laguerre polynomials are evaluated using explicit expressions

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::explicit_laguerre(scalar z, int d){
  scalar lval = 0.0;
  if ( d == 0 ){
    lval = 1.0;
  } else if ( d == 1 ){
    lval = 1.0 - z;
  } else if ( d == 2 ){
    lval = z*z - 4.0*z + 2.0;
    lval /= factorial(2);
  } else if ( d == 3 ){
    lval = -z*z*z + 9.0*z*z - 18.0*z + 6.0;
    lval /= factorial(3);
  } else if ( d == 4 ){
    lval = z*z*z*z - 16.0*z*z*z + 72.0*z*z - 96.0*z + 24.0;
    lval /= factorial(4);
  }
  return lval;
}

/**
   Laguerre polynomials are evaluated using recursive expressions

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::recursive_laguerre(scalar z, int d){
  scalar lval = 0.0;
  if ( d == 0 ) {
    lval = 1.0;
  } else if ( d == 1 ) {
    lval = 1.0 - z;
  } else {
    lval = ((scalar(2*d-1)-z)*recursive_laguerre(z,d-1) - scalar(d-1)*recursive_laguerre(z,d-2));
    lval /= scalar(d);
  }
  return lval;
}

/**
   Legendre polynomials are evaluated using explicit expressions

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::explicit_legendre(scalar z, int d){
  scalar pval = 0.0;
  if ( d == 0 ){
    pval = 1.0;
  } else if ( d == 1 ){
    pval = 2.0*z - 1.0;
  } else if ( d == 2 ){
    pval = 6.0*z*z - 6.0*z + 1.0;
  } else if ( d == 3 ){
    pval = 20.0*z*z*z - 30.0*z*z + 12.0*z - 1.0;
  } else if ( d == 4 ){
    pval = 70.0*z*z*z*z - 140.0*z*z*z + 90.0*z*z - 20.0*z + 1.0;
  }
  return pval;
}

/**
   Legendre polynomials are evaluated using general expressions

   @param z the point to evaluate the basis
   @param d the degree of basis function
*/
scalar OrthogonalPolynomials::general_legendre(scalar z, int d){
  scalar pval = 0.0;
  for (int k = 0; k <= d; k++){
    pval = pval + comb(d,k)*comb(d+k,k)*pow(-z,k);
  }
  pval = pval*pow(-1.0,d);
  return pval;
}

/**
   Function to evaluate polynomials
*/
void test_polynomials( int argc, char *argv[] ){

  OrthogonalPolynomials *poly = new OrthogonalPolynomials();

  scalar z = 1.1;
  int max_order = 10;
  int nruns = 100000;
  for (int j = 0; j < nruns; j++){
    // Test hermite polynomials
    printf("hermite\n");
    for (int i = 0; i < max_order; i++){
      printf("%2d %10.3f %10.3f \n", i, RealPart(poly->hermite(z,i)), RealPart(poly->unit_hermite(z,i)));
    }

    // Test Legendre polynomials
    printf("\nLegendre\n");
    for (int i = 0; i < max_order; i++){
      printf("%2d %10.3f %10.3f \n", i, RealPart(poly->legendre(z,i)), RealPart(poly->unit_legendre(z,i)));
    }

    // Test Laguerre polynomials
    printf("\nLaguerre\n");
    for (int i = 0; i < max_order; i++){
      printf("%2d %10.3f %10.3f \n", i, RealPart(poly->laguerre(z,i)), RealPart(poly->unit_laguerre(z,i)));
    }
  }
  delete poly;
}
