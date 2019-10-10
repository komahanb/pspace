#include<stdio.h>
#include<stdlib.h>
#include"BasisHelper.h"

// Constructor and Destructor
BasisHelper::BasisHelper(){}
BasisHelper::~BasisHelper(){}

/*
  Return the index set corresponding to maximum degrees of each parameter
*/
void BasisHelper::basisDegrees(const int nvars, const int *pmax, int **indx){
  if ( nvars == 1 ) {
    univariateBasisDegrees(pmax, indx);
  } else if ( nvars == 2 ) {
    bivariateBasisDegrees(pmax, indx);
  } else if ( nvars == 3 ) {
    trivariateBasisDegrees(pmax, indx);
  } else if ( nvars == 4 ) {
    quadvariateBasisDegrees(pmax, indx);
  } else if ( nvars == 5 ) {   
    pentavariateBasisDegrees(pmax, indx);
  } else {
    printf("Basis not implemented for more than five variables");
  }
}

/*
  Helps determine whether the evaluation is necessary as many terms
  are non-zero in jacobian matrix
*/
void BasisHelper::sparse(const int nvars, 
                         int *dmapi, int *dmapj, int *dmapk, 
                         bool *filter){
  for( int i = 0; i < nvars; i++ ){
    if (abs(dmapi[i] - dmapj[i]) <= dmapk[i]){
      filter[i] = true;
    } else {
      filter[i] = false;
    }
  }
}

void BasisHelper::univariateBasisDegrees(const int *pmax, int **indx){}
void BasisHelper::bivariateBasisDegrees(const int *pmax, int **indx){}
void BasisHelper::trivariateBasisDegrees(const int *pmax, int **indx){}
void BasisHelper::quadvariateBasisDegrees(const int *pmax, int **indx){}
void BasisHelper::pentavariateBasisDegrees(const int *pmax, int **indx){}
