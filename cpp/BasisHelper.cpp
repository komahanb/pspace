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
    univariateBasisDegrees(nvars, pmax, indx);
  } else if ( nvars == 2 ) {
    bivariateBasisDegrees(nvars, pmax, indx);
  } else if ( nvars == 3 ) {
    trivariateBasisDegrees(nvars, pmax, indx);
  } else if ( nvars == 4 ) {
    quadvariateBasisDegrees(nvars, pmax, indx);
  } else if ( nvars == 5 ) {   
    pentavariateBasisDegrees(nvars, pmax, indx);
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

void BasisHelper::univariateBasisDegrees(const int nvars, const int *pmax, int **indx){
  
  int nterms = 1;
  for(int i = 0; i < nvars; i++){
    nterms *= 1 + pmax[i];
  }

  int num_total_degrees = 1;
  for(int i = 0; i < nvars; i++){
    num_total_degrees += pmax[i];
  }

  // int *degree_list = new int[num_total_degrees];
  // for(int i = 0; i < nvars; i++){
  //   degree_list[i] = 
  // }
  
}

void BasisHelper::bivariateBasisDegrees(const int nvars, const int *pmax, int **indx){}
void BasisHelper::trivariateBasisDegrees(const int nvars, const int *pmax, int **indx){}
void BasisHelper::quadvariateBasisDegrees(const int nvars, const int *pmax, int **indx){}
void BasisHelper::pentavariateBasisDegrees(const int nvars, const int *pmax, int **indx){}
