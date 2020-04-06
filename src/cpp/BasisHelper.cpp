#include<stdio.h>
#include<map>

#include"BasisHelper.h"
#include"ArrayList.h"

using namespace std;

/**
   Constructor for helper class for creating orthonormal basis
   functions from univariate basis functions.

   @param basis_type the basis function type (tensor=0/complete=1)
*/
BasisHelper::BasisHelper( int _basis_type ){
  this->basis_type = _basis_type;
}

/**
   Destructor for BasisHelper class
*/
BasisHelper::~BasisHelper(){}

/**
  Return the degree index set and number of indices corresponding to
  maximum degrees of each parameter.

  @param nvars number of variables (parameters)
  @param pmax maximum degree of each variable (parameter)
  @param nbasis returns the number of basis entries
  @param basis_degrees returns the index set for each basis
*/
void BasisHelper::basisDegrees( const int nvars,
                                const int *pmax,
                                int *nbasis,
                                int **basis_degrees ){
  if ( nvars == 1 ) {
    uniVariateBasisDegrees(nvars, pmax, nbasis, basis_degrees);
  } else if ( nvars == 2 ) {
    if (this->basis_type == 0) {
      biVariateBasisDegreesTensor(nvars, pmax, nbasis, basis_degrees);
    } else {
      biVariateBasisDegreesComplete(nvars, pmax, nbasis, basis_degrees);
    }
  } else if ( nvars == 3 ) {
    if (this->basis_type == 0) {
      triVariateBasisDegreesTensor(nvars, pmax, nbasis, basis_degrees);
    } else {
      triVariateBasisDegreesComplete(nvars, pmax, nbasis, basis_degrees);
    }
  } else if ( nvars == 4 ) {
    if (this->basis_type == 0) {
      quadVariateBasisDegreesTensor(nvars, pmax, nbasis, basis_degrees);
    } else {
      quadVariateBasisDegreesComplete(nvars, pmax, nbasis, basis_degrees);
    }
  } else if ( nvars == 5 ) {
    if (this->basis_type == 0) {
      pentaVariateBasisDegreesTensor(nvars, pmax, nbasis, basis_degrees);
    } else {
      pentaVariateBasisDegreesComplete(nvars, pmax, nbasis, basis_degrees);
    }
  } else {
    printf("Multivariate basis construction is not implemented for %d variables\n", nvars);
  }
}

/**
  Helps determine whether the evaluation is necessary as many terms
  are non-zero in jacobian matrix

  @param dmapi degree of i-th entry of jacobian
  @param dmapj degree of j-th entry of jacobian
  @param dmapf degree of integrand
*/
void BasisHelper::sparse( const int nvars,
                          int *dmapi,
                          int *dmapj,
                          int *dmapk,
                          bool *filter ){
  for( int i = 0; i < nvars; i++ ){
    if (abs(dmapi[i] - dmapj[i]) <= dmapk[i]){
      filter[i] = true;
    } else {
      filter[i] = false;
    }
  }
}

/**
   Function that constructs a uni-variate basis

   @param nvars number of variables (parameters)
   @param pmax maximum degree of each variable (parameter)
   @param nbasis returns the number of basis entries
   @param basis_degrees returns the index set for each basis
*/
void BasisHelper::uniVariateBasisDegrees( const int nvars,
                                          const int *pmax,
                                          int *nbasis,
                                          int **basis_degrees ){
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
    dmap[k]->getEntries(&basis_degrees[ctr]);
    ctr = ctr + nrecords;
  }
  nbasis[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

/**
   Function that constructs a bi-variate basis

   @param nvars number of variables (parameters)
   @param pmax maximum degree of each variable (parameter)
   @param nbasis returns the number of basis entries
   @param basis_degrees returns the index set for each basis
*/
void BasisHelper::biVariateBasisDegreesTensor( const int nvars,
                                               const int *pmax,
                                               int *nbasis,
                                               int **basis_degrees ){
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
    dmap[k]->getEntries(&basis_degrees[ctr]);
    ctr = ctr + nrecords;
  }
  nbasis[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

/**
   Function that constructs a tri-variate basis

   @param nvars number of variables (parameters)
   @param pmax maximum degree of each variable (parameter)
   @param nbasis returns the number of basis entries
   @param basis_degrees returns the index set for each basis
*/
void BasisHelper::triVariateBasisDegreesTensor( const int nvars,
                                                const int *pmax,
                                                int *nbasis,
                                                int **basis_degrees ){
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
    dmap[k]->getEntries(&basis_degrees[ctr]);
    ctr = ctr + nrecords;
  }
  nbasis[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

/**
   Function that constructs a quadVariate basis

   @param nvars number of variables (parameters)
   @param pmax maximum degree of each variable (parameter)
   @param nbasis returns the number of basis entries
   @param basis_degrees returns the index set for each basis
*/
void BasisHelper::quadVariateBasisDegreesTensor( const int nvars,
                                                 const int *pmax,
                                                 int *nbasis,
                                                 int **basis_degrees ){
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
    dmap[k]->getEntries(&basis_degrees[ctr]);
    ctr = ctr + nrecords;
  }
  nbasis[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

/**
   Function that constructs a penta-variate basis using tensor product
   rule.

   @param nvars number of variables (parameters)
   @param pmax maximum degree of each variable (parameter)
   @param nbasis returns the number of basis entries
   @param basis_degrees returns the index set for each basis
*/
void BasisHelper::pentaVariateBasisDegreesTensor( const int nvars,
                                                  const int *pmax,
                                                  int *nbasis,
                                                  int **basis_degrees ){
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
    dmap[k]->getEntries(&basis_degrees[ctr]);
    ctr = ctr + nrecords;
  }
  nbasis[0] = ctr;

  // Delete the array lists
  for (int k = 0; k < num_total_degrees; k++){
    delete dmap[k];
  }
}

/**
   Compute the factorial of an integer

   @param n integer number for factorial
*/
int factorial(const int n) {
  unsigned long long fact = 1;
  for(int i = 1; i <=n; ++i){
    fact *= i;
  }
  return (int) fact;
}

/**
   Returns the maximum of an array

   @param n length of array
   @param vals values
 */
int maxval(const int n, const int *vals){
  int mval = vals[0];
  for(int i = 1; i < n; i++){
    if (vals[i] > mval) {
      mval = vals[i];
    }
  }
  return mval;
}

/**
   Creates bi-variate basis using complete polynomial rule

   @param nvars number of variables (parameters)
   @param pmax maximum degree of each variable (parameter)
   @param nbasis returns the number of basis entries
   @param basis_degrees returns the index set for each basis
*/
void BasisHelper::biVariateBasisDegreesComplete( const int nvars,
                                                 const int *pmax,
                                                 int *nbasis,
                                                 int **basis_degrees ){
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

  // Fill the basis_degrees array with basis entries of increasing degree
  int ctr = 0;
  for (int k = 0; k <= maxpmax; k++){
    int nbasis = dmap[k]->getNumEntries(); // of degree k
    dmap[k]->getEntries(&basis_degrees[ctr]);
    ctr = ctr + nbasis; // add nbasis
  }
  nbasis[0] = ctr;

  // Delete the array lists
  for (int k = 0; k <= maxpmax; k++){
    delete dmap[k];
  }
}

/**
   Creates tri-variate basis using complete polynomial rule

   @param nvars number of variables (parameters)
   @param pmax maximum degree of each variable (parameter)
   @param nbasis returns the number of basis entries
   @param basis_degrees returns the index set for each basis
*/
void BasisHelper::triVariateBasisDegreesComplete( const int nvars,
                                                  const int *pmax,
                                                  int *nbasis,
                                                  int **basis_degrees ){
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
    dmap[k]->getEntries(&basis_degrees[ctr]);
    ctr = ctr + nrecords;
  }
  nbasis[0] = ctr;

  // Delete the array lists
  for (int k = 0; k <= maxpmax; k++){
    delete dmap[k];
  }
}

/**
   Creates quad-variate basis using complete polynomial rule

   @param nvars number of variables (parameters)
   @param pmax maximum degree of each variable (parameter)
   @param nbasis returns the number of basis entries
   @param basis_degrees returns the index set for each basis
*/
void BasisHelper::quadVariateBasisDegreesComplete( const int nvars,
                                                   const int *pmax,
                                                   int *nbasis,
                                                   int **basis_degrees ){
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
    dmap[k]->getEntries(&basis_degrees[ctr]);
    ctr = ctr + nrecords;
  }
  nbasis[0] = ctr;

  // Delete the array lists
  for (int k = 0; k <= maxpmax; k++){
    delete dmap[k];
  }
}

/**
   Creates pentaVariate basis using complete polynomial rule

   @param nvars number of variables (parameters)
   @param pmax maximum degree of each variable (parameter)
   @param nbasis returns the number of basis entries
   @param basis_degrees returns the index set for each basis
*/
void BasisHelper::pentaVariateBasisDegreesComplete( const int nvars,
                                                    const int *pmax,
                                                    int *nbasis,
                                                    int **basis_degrees ){
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
    dmap[k]->getEntries(&basis_degrees[ctr]);
    ctr = ctr + nrecords;
  }
  nbasis[0] = ctr;

  // Delete the array lists
  for (int k = 0; k <= maxpmax; k++){
    delete dmap[k];
  }
}
