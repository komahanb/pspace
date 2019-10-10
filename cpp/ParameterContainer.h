#include<stdio.h>
#include"AbstractParameter.h"
#include"BasisHelper.h"
#include"QuadratureHelper.h"

class ParameterContainer {
 public:
  // Constructor and Destructor
  ParameterContainer();
  ~ParameterContainer();
  
  // Key funtionalities
  void addParameter();
  void basis();
  void quadrature();

  // Accessors
  void getNumBasisTerms();
  void getNumQuadraturePoints();

  // Initiliazation tasks
  void initializeBasis();
  void initializeQuadrature();

 private:
  AbstractParameter *plist; // use vector?  
  int num_parameters;
  int *param_max_deg;
  int **dindex;
  double **Z, **Y, **W;
  int num_basis_terms;
  int num_quadrature_points; 

  // Private Functions
  // void basisTerm();
  // void basisGivenDegrees();
}
