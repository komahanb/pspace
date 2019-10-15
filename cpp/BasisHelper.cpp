#include<stdio.h>
#include<stdlib.h>
#include<map>
#include<list>

#include"BasisHelper.h"

using namespace std;

// Constructor and Destructor
BasisHelper::BasisHelper(){}
BasisHelper::~BasisHelper(){}

/*
  Return the index set corresponding to maximum degrees of each parameter
*/
void BasisHelper::basisDegrees(const int nvars, const int *pmax, 
                               int *nindices, int **indx){
  if ( nvars == 1 ) {
    univariateBasisDegrees(nvars, pmax, nindices, indx);
  } else if ( nvars == 2 ) {
    bivariateBasisDegrees(nvars, pmax, nindices, indx);
  } else if ( nvars == 3 ) {
    trivariateBasisDegrees(nvars, pmax, nindices, indx);
  } else if ( nvars == 4 ) {
    quadvariateBasisDegrees(nvars, pmax, nindices, indx);
  } else if ( nvars == 5 ) {   
    pentavariateBasisDegrees(nvars, pmax, nindices, indx);
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

void BasisHelper::univariateBasisDegrees(const int nvars, const int *pmax, 
                                         int *nindices, int **indx){
  // Number of terms from tensor product
  int nterms = 1;
  for (int i = 0; i < nvars; i++){
    nterms *= 1 + pmax[i];
  }

  int num_total_degrees = 1;
  for (int i = 0; i < nvars; i++){
    num_total_degrees += pmax[i];
  }

  // Add indices degreewise
  list<int*> **degree_list = new list<int*>*[num_total_degrees];
  for (int ii = 0; ii <= pmax[0]; ii++){
    int tmp[nvars];
    tmp[0] = ii;
    degree_list[ii] = new list<int*>[nterms];
  }

  // Flatten the list with ascending degrees
  for (int k = 0; k < num_total_degrees; k++){
    
  }
  // https://www.geeksforgeeks.org/list-cpp-stl/
  int d1[nvars];
  int d2[nvars];

  // degree_list->push_back(d1);
  // degree_list->push_back(d2);

  // for (int i = 0; i < num_total_degrees; i++){    
  //   // Get the number of entries
  //   int ndegentries = 4;
  //   for (int j = 0; j < ndegentries; j++){
  //     kdegs = map[j]
  //     for (int p = 0; p < nvars; p++){
  //       indx[p][i] = kdegs[p];
  //     }
  //   }
  // }

}

void BasisHelper::bivariateBasisDegrees(const int nvars, const int *pmax, 
                                        int *nindices, int **indx){}
void BasisHelper::trivariateBasisDegrees(const int nvars, const int *pmax, 
                                         int *nindices, int **indx){}
void BasisHelper::quadvariateBasisDegrees(const int nvars, const int *pmax, 
                                          int *nindices, int **indx){}
void BasisHelper::pentavariateBasisDegrees(const int nvars, const int *pmax, 
                                           int *nindices, int **indx){}
