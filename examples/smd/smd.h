#include "TACSElement.h"
class SMD : public TACSElement{  
 public:
  SMD(double m, double c, double k);
  void addResidual( double time, TacsScalar res[],
                    const TacsScalar Xpts[],
                    const TacsScalar vars[],
                    const TacsScalar dvars[],
                    const TacsScalar ddvars[] );
  void getInitConditions( TacsScalar vars[],
                          TacsScalar dvars[],
                          TacsScalar ddvars[],
                          const TacsScalar Xpts[] );
  int numDisplacements(){
    return 1;
  }; 

  int numNodes() {
    return 1;
  }

 private:
  double m, c, k;
};
