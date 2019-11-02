#include "TACSElement.h"
class PPA : public TACSElement{  
 public:
  PPA( double xcm, double xf, 
      double m  , double If, 
      double ch , double ca,
      double kh , double ka );
  void addResidual( double time, TacsScalar res[],
                    const TacsScalar Xpts[],
                    const TacsScalar vars[],
                    const TacsScalar dvars[],
                    const TacsScalar ddvars[] );
  void getInitConditions( TacsScalar vars[],
                          TacsScalar dvars[],
                          TacsScalar ddvars[],
                          const TacsScalar Xpts[] );
  void addJacobian( double time, TacsScalar J[],
                    double alpha, double beta, double gamma,
                    const TacsScalar X[],
                    const TacsScalar v[],
                    const TacsScalar dv[],
                    const TacsScalar ddv[] );
  int numDisplacements(){
    return 2;
  };  
  int numNodes() {
    return 1;
  }
  const char * elementName(){
    return "PPA";
  }
  double xcm, xf, m, If, ch, ca, kh, ka, s;
};
