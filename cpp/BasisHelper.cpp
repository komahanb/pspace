#include<stdio.h>
#include<stdlib.h>
#include<map>
#include<list>
#include<vector>

#include"BasisHelper.h"
#include"ArrayList.h"

using namespace std;

// Constructor
BasisHelper::BasisHelper(int _basis_type){
  this->basis_type = _basis_type;
}

// Destructor
BasisHelper::~BasisHelper(){}

/*
  Return the index set corresponding to maximum degrees of each parameter
*/
void BasisHelper::basisDegrees(const int nvars, const int *pmax, 
                               int *nindices, int **indx){
  if ( nvars == 1 ) {
    univariateBasisDegrees(nvars, pmax, nindices, indx);
  } else if ( nvars == 2 ) {
    if (this->basis_type == 0) {
      bivariateBasisDegrees(nvars, pmax, nindices, indx);
    } else {
      bivariateBasisDegreesComplete(nvars, pmax, nindices, indx);
    } 
  } else if ( nvars == 3 ) {
    if (this->basis_type == 0) {
      trivariateBasisDegrees(nvars, pmax, nindices, indx);
    } else {
      trivariateBasisDegreesComplete(nvars, pmax, nindices, indx);
    } 
  } else if ( nvars == 4 ) {
    if (this->basis_type == 0) {
      quadvariateBasisDegrees(nvars, pmax, nindices, indx);
    } else {
      quadvariateBasisDegreesComplete(nvars, pmax, nindices, indx);
    }    
  } else if ( nvars == 5 ) {   
    if (this->basis_type == 0) {
      pentavariateBasisDegrees(nvars, pmax, nindices, indx);
    } else {
      pentavariateBasisDegreesComplete(nvars, pmax, nindices, indx);
    }
  } else {
    printf("Basis not implemented for %d variables\n", nvars);
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

  // Create a map of empty array lists
  std::map<int,ArrayList*> dmap;
  for (int k = 0; k < num_total_degrees; k++){
    dmap.insert(pair<int, ArrayList*>(k, new ArrayList(nterms, nvars)));
  }

  // Add degree wise tuples into each arraylist
  for (int ii = 0; ii <= pmax[0]; ii++){
    int tuple[] = {ii};
    dmap[ii]->addEntry(tuple);
  }

  int ctr = 0;
  for (int k = 0; k < num_total_degrees; k++){    
    int nrecords = dmap[k]->getNumEntries();
    dmap[k]->getEntries(&indx[ctr]);
    ctr = ctr + nrecords;
  }  
  nindices[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

void BasisHelper::bivariateBasisDegrees(const int nvars, const int *pmax, 
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

  // Create a map of empty array lists
  std::map<int,ArrayList*> dmap;
  for (int k = 0; k < num_total_degrees; k++){
    dmap.insert(pair<int, ArrayList*>(k, new ArrayList(nterms, nvars)));
  }

  // Add degree wise tuples into each arraylist
  for (int ii = 0; ii <= pmax[0]; ii++){
    for (int jj = 0; jj <= pmax[1]; jj++){
      int tuple[] = {ii,jj};
      dmap[ii+jj]->addEntry(tuple);
    }
  }

  int ctr = 0;
  for (int k = 0; k < num_total_degrees; k++){    
    int nrecords = dmap[k]->getNumEntries();
    dmap[k]->getEntries(&indx[ctr]);
    ctr = ctr + nrecords;
  }  
  nindices[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

void BasisHelper::trivariateBasisDegrees(const int nvars, const int *pmax, 
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

  // Create a map of empty array lists
  std::map<int,ArrayList*> dmap;
  for (int k = 0; k < num_total_degrees; k++){
    dmap.insert(pair<int, ArrayList*>(k, new ArrayList(nterms, nvars)));
  }

  // Add degree wise tuples into each arraylist
  for (int ii = 0; ii <= pmax[0]; ii++){
    for (int jj = 0; jj <= pmax[1]; jj++){
      for (int kk = 0; kk <= pmax[2]; kk++){
        int tuple[] = {ii,jj,kk};
        dmap[ii+jj+kk]->addEntry(tuple);
      }
    }
  }

  int ctr = 0;
  for (int k = 0; k < num_total_degrees; k++){    
    int nrecords = dmap[k]->getNumEntries();
    dmap[k]->getEntries(&indx[ctr]);
    ctr = ctr + nrecords;
  }  
  nindices[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

void BasisHelper::quadvariateBasisDegrees(const int nvars, const int *pmax, 
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

  // Create a map of empty array lists
  std::map<int,ArrayList*> dmap;
  for (int k = 0; k < num_total_degrees; k++){
    dmap.insert(pair<int, ArrayList*>(k, new ArrayList(nterms, nvars)));
  }

  // Add degree wise tuples into each arraylist
  for (int ii = 0; ii <= pmax[0]; ii++){
    for (int jj = 0; jj <= pmax[1]; jj++){
      for (int kk = 0; kk <= pmax[2]; kk++){
        for (int ll = 0; ll <= pmax[3]; ll++){
          int tuple[] = {ii,jj,kk,ll};
          dmap[ii+jj+kk+ll]->addEntry(tuple);
        }
      }
    }
  }

  int ctr = 0;
  for (int k = 0; k < num_total_degrees; k++){    
    int nrecords = dmap[k]->getNumEntries();
    dmap[k]->getEntries(&indx[ctr]);
    ctr = ctr + nrecords;
  }  
  nindices[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

void BasisHelper::pentavariateBasisDegrees(const int nvars, const int *pmax, 
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

  // Create a map of empty array lists
  std::map<int,ArrayList*> dmap;
  for (int k = 0; k < num_total_degrees; k++){
    dmap.insert(pair<int, ArrayList*>(k, new ArrayList(nterms, nvars)));
  }

  // Add degree wise tuples into each arraylist
  for (int ii = 0; ii <= pmax[0]; ii++){
    for (int jj = 0; jj <= pmax[1]; jj++){
      for (int kk = 0; kk <= pmax[2]; kk++){
        for (int ll = 0; ll <= pmax[3]; ll++){
          for (int mm = 0; mm <= pmax[4]; mm++){
            int tuple[] = {ii,jj,kk,ll,mm};
            dmap[ii+jj+kk+ll+mm]->addEntry(tuple);
          }
        }
      }
    }
  }

  int ctr = 0;
  for (int k = 0; k < num_total_degrees; k++){    
    int nrecords = dmap[k]->getNumEntries();
    dmap[k]->getEntries(&indx[ctr]);
    ctr = ctr + nrecords;
  }  
  nindices[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

// Compute the factorial of an integer
int factorial(const int n) {
  unsigned long long fact = 1;
  for(int i = 1; i <=n; ++i){
    fact *= i;
  }
  return (int) fact;
}

int maxval(const int n, const int *vals){
  int mval = vals[0];
  for(int i = 1; i < n; i++){
    if (vals[i] > mval) {
      mval = vals[i];
    }
  }
  return mval;
}

void BasisHelper::bivariateBasisDegreesComplete(const int nvars, const int *pmax, 
                                                int *nindices, int **indx){
  // Number of terms from factorials
  int maxpmax = maxval(nvars, pmax);
  int nterms = factorial(nvars+maxpmax)/(factorial(nvars)*factorial(maxpmax));
  
  // Create a map of empty array lists
  std::map<int,ArrayList*> dmap;
  for (int k = 0; k <= maxpmax; k++){
    dmap.insert(pair<int, ArrayList*>(k, new ArrayList(nterms, nvars)));    
  }

  // Add degree wise tuples into each arraylist
  for (int ii = 0; ii <= pmax[0]; ii++){
    for (int jj = 0; jj <= pmax[1]; jj++){
      int dsum = ii+jj;
      if (dsum <= maxpmax){
        int tuple[] = {ii,jj}; // nvars
        dmap[dsum]->addEntry(tuple);
      }
    }
  }

  // Fill the indx array with basis entries of increasing degree
  int ctr = 0;
  for (int k = 0; k <= maxpmax; k++){  
    int nbasis = dmap[k]->getNumEntries(); // of degree k
    dmap[k]->getEntries(&indx[ctr]);
    ctr = ctr + nbasis; // add nbasis
  }  
  nindices[0] = ctr;

  // Delete the array lists
  for (int k = 0; k <= maxpmax; k++){
    delete dmap[k];
  }
}

void BasisHelper::trivariateBasisDegreesComplete(const int nvars, const int *pmax, 
                                                 int *nindices, int **indx){
  // Number of terms from factorials
  int maxpmax = maxval(nvars, pmax);
  int nterms = factorial(nvars+maxpmax)/(factorial(nvars)*factorial(maxpmax));

  // Create a map of empty array lists
  std::map<int,ArrayList*> dmap;
  for (int k = 0; k <= maxpmax; k++){
    dmap.insert(pair<int, ArrayList*>(k, new ArrayList(nterms, nvars)));
  }

  // Add degree wise tuples into each arraylist
  for (int ii = 0; ii <= pmax[0]; ii++){
    for (int jj = 0; jj <= pmax[1]; jj++){
      for (int kk = 0; kk <= pmax[2]; kk++){
        int dsum = ii+jj+kk;
        if (dsum <= maxpmax){
          int tuple[] = {ii,jj,kk};
          dmap[dsum]->addEntry(tuple);
        }
      }
    }
  }

  int ctr = 0;
  for (int k = 0; k <= maxpmax; k++){   
    int nrecords = dmap[k]->getNumEntries();
    dmap[k]->getEntries(&indx[ctr]);
    ctr = ctr + nrecords;
  }  
  nindices[0] = ctr;

  // Delete the array lists
  for (int k = 0; k <= maxpmax; k++){
    delete dmap[k];
  }
}

void BasisHelper::quadvariateBasisDegreesComplete(const int nvars, const int *pmax, 
                                                  int *nindices, int **indx){
  // Number of terms from factorials
  int maxpmax = maxval(nvars, pmax);
  int nterms = factorial(nvars+maxpmax)/(factorial(nvars)*factorial(maxpmax));

  // Create a map of empty array lists
  std::map<int,ArrayList*> dmap;
  for (int k = 0; k <= maxpmax; k++){
    dmap.insert(pair<int, ArrayList*>(k, new ArrayList(nterms, nvars)));
  }

  // Add degree wise tuples into each arraylist
  for (int ii = 0; ii <= pmax[0]; ii++){
    for (int jj = 0; jj <= pmax[1]; jj++){
      for (int kk = 0; kk <= pmax[2]; kk++){
        for (int ll = 0; ll <= pmax[3]; ll++){
          int dsum = ii+jj+kk+ll;
          if (dsum <= maxpmax){
            int tuple[] = {ii,jj,kk,ll};
            dmap[dsum]->addEntry(tuple);
          }
        }
      }
    }
  }

  int ctr = 0;
  for (int k = 0; k <= maxpmax; k++){    
    int nrecords = dmap[k]->getNumEntries();
    dmap[k]->getEntries(&indx[ctr]);
    ctr = ctr + nrecords;
  }
  nindices[0] = ctr;

  // Delete the array lists
  for (int k = 0; k <= maxpmax; k++){
    delete dmap[k];
  }
}

void BasisHelper::pentavariateBasisDegreesComplete(const int nvars, const int *pmax, 
                                                   int *nindices, int **indx){
  // Number of terms from factorials
  int maxpmax = maxval(nvars, pmax);
  int nterms = factorial(nvars+maxpmax)/(factorial(nvars)*factorial(maxpmax));
  
  // Create a map of empty array lists
  std::map<int,ArrayList*> dmap;
  for (int k = 0; k <= maxpmax; k++){
    dmap.insert(pair<int, ArrayList*>(k, new ArrayList(nterms, nvars)));
  }

  // Add degree wise tuples into each arraylist
  for (int ii = 0; ii <= pmax[0]; ii++){
    for (int jj = 0; jj <= pmax[1]; jj++){
      for (int kk = 0; kk <= pmax[2]; kk++){
        for (int ll = 0; ll <= pmax[3]; ll++){
          for (int mm = 0; mm <= pmax[4]; mm++){
            int dsum = ii+jj+kk+ll+mm;
            if (dsum <= maxpmax){
              int tuple[] = {ii,jj,kk,ll,mm};
              dmap[dsum]->addEntry(tuple);
            }
          }
        }
      }
    }
  }

  int ctr = 0;
  for (int k = 0; k <= maxpmax; k++){   
    int nrecords = dmap[k]->getNumEntries();
    dmap[k]->getEntries(&indx[ctr]);
    ctr = ctr + nrecords;
  }  
  nindices[0] = ctr;

  // Delete the array lists
  for (int k = 0; k <= maxpmax; k++){
    delete dmap[k];
  }
}
