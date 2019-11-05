/*
  Class to evaluate orthogonal polynomials.

  Author: Komahan Boopathy (komahanboopathy@gmail.com)
*/

#include <stdio.h>
#include <math.h>

#include "OrthogonalPolynomials.h"

//===================================================================//
// Constructor and Destructor
//===================================================================//

/*
  Constructor
*/
OrthogonalPolynomials::OrthogonalPolynomials(){}

/*
  Destructor
*/
OrthogonalPolynomials::~OrthogonalPolynomials(){}

//===================================================================//
// Public functions
//===================================================================//

/*  
    Evaluate Hermite polynomials
*/
double OrthogonalPolynomials::hermite(double z, int d){
  if ( d <= 4 ) {
    return explicit_hermite(z,d);
  } else {
    return recursive_hermite(z,d);
  }
}

/*  
    Evaluate unit hermite polynomials
*/
double OrthogonalPolynomials::unit_hermite(double z, int d){
  return hermite(z,d)/sqrt(factorial(d));
}

/*  
    Evaluate Laguerre polynomials
*/
double OrthogonalPolynomials::laguerre(double z, int d){
  if ( d <= 4 ) {
    return explicit_laguerre(z,d);
  } else {
    return recursive_laguerre(z,d);
  }
}

/*  
    Evaluate unit Laguerre polynomials (already normalized)
*/
double OrthogonalPolynomials::unit_laguerre(double z, int d){
  return laguerre(z,d);
}

/*  
    Evaluate Legendre polynomials
*/
double OrthogonalPolynomials::legendre(double z, int d){
  if ( d <= 4 ) {
    return explicit_legendre(z,d);
  } else {
    return general_legendre(z,d);
  }
}

/*  
    Evaluate unit legendre polynomials
*/
double OrthogonalPolynomials::unit_legendre(double z, int d){
  return legendre(z,d)*sqrt(double(2*d+1));
}

//===================================================================//
// Private functions
//===================================================================//

/*
  Combination nCr = n!/((n-r)!r!)
*/
double OrthogonalPolynomials::comb(int n, int r){
  double nfact  = factorial(n);
  double rfact  = factorial(r);
  double nrfact = factorial(n-r);
  return nfact/(rfact*nrfact);
}

/*
  Compute the factorial of a number
 */
double OrthogonalPolynomials::factorial( int n ){
  double factorial;
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
      factorial *= double(i);
    }
  }
  return factorial;
}

/*  
    Hermite polynomials are evaluated using explicit expressions
*/
double OrthogonalPolynomials::explicit_hermite(double z, int d){
  double hval = 0.0;
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

/*  
    Hermite polynomials are evaluated using recursive expressions
*/
double OrthogonalPolynomials::recursive_hermite(double z, int d){
  double hval = 0.0;
  if ( d == 0 ) {
    hval = 1.0;
  } else if ( d == 1 ) {
    hval = z;
  } else {
    hval = z*recursive_hermite(z,d-1) - double(d-1)*recursive_hermite(z,d-2);
  }
  return hval;
}

/*  
    Laguerre polynomials are evaluated using explicit expressions
*/
double OrthogonalPolynomials::explicit_laguerre(double z, int d){
  double lval = 0.0;
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

/*  
    Laguerre polynomials are evaluated using recursive expressions
*/
double OrthogonalPolynomials::recursive_laguerre(double z, int d){
  double lval = 0.0;
  if ( d == 0 ) {
    lval = 1.0;
  } else if ( d == 1 ) {
    lval = 1.0 - z;
  } else {    
    lval = ((double(2*d-1)-z)*recursive_laguerre(z,d-1) - double(d-1)*recursive_laguerre(z,d-2));
    lval /= double(d);
  }
  return lval;
}

/*  
    Legendre polynomials are evaluated using explicit expressions
*/
double OrthogonalPolynomials::explicit_legendre(double z, int d){
  double pval = 0.0;
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

/*  
    Legendre polynomials are evaluated using general expressions
*/
double OrthogonalPolynomials::general_legendre(double z, int d){
  double pval = 0.0;
  for (int k = 0; k <= d; k++){
    pval = pval + comb(d,k)*comb(d+k,k)*pow(-z,k);
  }
  pval = pval*pow(-1.0,d);
  return pval;
}

void test_polynomials( int argc, char *argv[] ){
  
  OrthogonalPolynomials *poly = new OrthogonalPolynomials();

  double z = 1.1;
  int max_order = 10;
  int nruns = 100000;  
  for (int j = 0; j < nruns; j++){
    // Test hermite polynomials
    printf("hermite\n");
    for (int i = 0; i < max_order; i++){
      printf("%2d %10.3f %10.3f \n", i, poly->hermite(z,i), poly->unit_hermite(z,i));
    }

    // Test Legendre polynomials
    printf("\nLegendre\n");
    for (int i = 0; i < max_order; i++){
      printf("%2d %10.3f %10.3f \n", i, poly->legendre(z,i), poly->unit_legendre(z,i));
    }

    // Test Laguerre polynomials
    printf("\nLaguerre\n");
    for (int i = 0; i < max_order; i++){
      printf("%2d %10.3f %10.3f \n", i, poly->laguerre(z,i), poly->unit_laguerre(z,i));
    }
  }
  delete poly;
}

