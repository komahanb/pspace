#ifndef FOUR_BAR_PROBLEM
#define FOUR_BAR_PROBLEM

#include "scalar.h"

class FourBarProblem {
 public:
  void getObjCon(TacsScalar *fvals);
  void getGradient();
 private:
  int numvars;
  int numcons;
};

#endif
