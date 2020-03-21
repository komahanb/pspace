#ifndef PARAMETER_CONTAINER
#define PARAMETER_CONTAINER

#include<stdio.h>
#include<map>

#include"AbstractParameter.h"
#include"BasisHelper.h"
#include"QuadratureHelper.h"

class ParameterContainer {
 public:
  // Constructor and Destructor
  ParameterContainer(int basis_type=0, int quadrature_type=0);
  ~ParameterContainer();
  
  // Key funtionalities
  void addParameter(AbstractParameter *param);
  // void addParameter(AbstractParameter *param, int max_deg, );

  // Evaluate basis at quadrature points
  scalar quadrature(int q, scalar *zq, scalar *yq);
  scalar basis(int k, scalar *z);

  // Accessors
  int getNumBasisTerms();
  int getNumParameters();
  int getNumQuadraturePoints();
  void getBasisParamDeg(int k, int *degs);
  void getBasisParamMaxDeg(int *pmax);

  // Initiliazation tasks
  void initialize();
  void initializeBasis(const int *pmax);
  void initializeQuadrature(const int *nqpts);
  
 private:
  // Maintain a map of parameters
  std::map<int,AbstractParameter*> pmap;

  int tnum_parameters;        // total number of parameters
  int tnum_basis_terms;       // total number of basis terms
  int tnum_quadrature_points; // total number of quadrature points

  int *param_max_degree;   // maximum monomial degree of each parameter
  int **dindex;         // parameterwise degree for each basis entry
  scalar **Z, **Y, *W; 

  // Helpers to access basis evaluation and quadrature points
  BasisHelper *bhelper;
  QuadratureHelper *qhelper;
};
#endif
