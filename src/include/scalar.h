#ifndef COMPLEX_STEP_H
#define COMPLEX_STEP_H

#include <stdlib.h>
#include <complex>

/**
  Use the complex type
*/
using namespace std;

typedef std::complex<double> Complex;
typedef double Real;

/**
  Define the basic scalar type
*/
#ifdef USE_COMPLEX
#define MPI_TYPE MPI_DOUBLE_COMPLEX
typedef Complex scalar;
#else
#define MPI_TYPE MPI_DOUBLE
typedef double scalar;
#endif

/**
   Define the real part function for the complex data type
*/
inline double RealPart( const std::complex<double>& c ){
  return real(c);
}

/**
   Define the imaginary part function for the complex data type
*/
inline double ImagPart( const std::complex<double>& c ){
  return imag(c);
}

/**
   Dummy function for real part
*/
inline double RealPart( const double& r ){
  return r;
}

/**
   Compute the absolute value
*/
inline std::complex<double> dabs( const std::complex<double>& c ){
  if (real(c) < 0.0){
    return -c;
  }
  return c;
}

#endif
