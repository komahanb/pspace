#include<stdio.h>
#include<map>

#include"AbstractParameter.h"
#include"BasisHelper.h"
#include"QuadratureHelper.h"

using namespace std;

class ParameterContainer {
 public:
  // Constructor and Destructor
  ParameterContainer();
  ~ParameterContainer();
  
  // Key funtionalities
  void addParameter(AbstractParameter *param);
  //void basis();
  // void quadrature();

  // Accessors
  int getNumBasisTerms();
  int getNumParameters();
  int getNumQuadraturePoints();

  // Initiliazation tasks
  void initializeBasis();
  void initializeQuadrature();

 private:

  // Maintain a map of parameters
  std::map<int,AbstractParameter*> pmap;

  int num_parameters;
  int num_basis_terms;
  int num_quadrature_points; 

  int *param_max_deg;
  int **dindex;
  double **Z, **Y, **W;

  BasisHelper *bhelper;
  QuadratureHelper *qhelper;
  
  // Private Functions
  // void basisTerm();
  // void basisGivenDegrees();
};
