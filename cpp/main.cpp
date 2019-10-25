#include <string>
#include <map>

#include"ParameterContainer.h"
#include"ParameterFactory.h"

class Object{
};

// class ParamData {
//   int pid;
//   int deg;
//   Object *elem;
//   void (*set)(Object*, double);
//   double (*get)(Object*);  
// };

// Mass test element
class Mass : public Object {
public:
  void setMass(double m){
    this->m = m;
  };
  double getMass(){
    return this->m;
  };

  static double getMass(void* obj){
    Mass *self = (Mass*) obj;
    return self->m;
  };

  static void setMass(void* obj, double mass){
    Mass *self = (Mass*) obj;
    self->m = mass;
  };

private:
  double m;  
};

// Spring test element
class Spring : public Object {
public:
  static double getStiffness(void* obj){
    Spring *self = (Spring*) obj;
    return self->k;
  };
  static void setStiffness(void* obj, double stiff){
    Spring *self = (Spring*) obj;
    self->k = stiff;
  };

private:
  double k;  
};

void hello(int a){
  printf("integer number is %d \n",a);
}

void hello2(double a){
  printf("double number is %f \n", a);
}

void setParameterID(AbstractParameter *param, int pid){
  param->setParameterID(pid);
}

int getParameterID(AbstractParameter *param){
  return param->getParameterID();
}

int main( int argc, char *argv[] ){

  Mass *m1, *m2, *m3;
  Spring *s1, s2;

  // Create random parameters
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *p1 = factory->createNormalParameter(-4.0, 0.5);
  //p1->elem = m1;
  p1->setClient(m1);
  // p1->set = Mass::setMass;
  p1->setClientFunction(Mass::setMass);
  
  p1->get = Mass::getMass;
  
  AbstractParameter *p2 = factory->createUniformParameter(-5.0, 4.0);  
  AbstractParameter *p3 = factory->createExponentialParameter(6.0, 1.0);
  AbstractParameter *p4 = factory->createExponentialParameter(6.0, 1.0);
  AbstractParameter *p5 = factory->createNormalParameter(-4.0, 0.5);  

  // Create container and add random paramters
  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(p1);
  pc->addParameter(p2);
  pc->addParameter(p3);
  pc->addParameter(p4);
  pc->addParameter(p5);

  // p1->set(p1->elem, 2.231324);
  p1->updateValue(2.231324);
  printf("%f\n", p1->getValue());

  // Set max degrees of expansion and get corresponding number of
  // quadrature points
  const int nvars = pc->getNumParameters();
  // int *pmax = new int[nvars];
  // int *nqpts = new int[nvars];
  // for (int i = 0; i < nvars; i++){
  //   pmax[i] = i+2;
  //   nqpts[i] = pmax[i]+1;
  // }

  // printf("%d \n", p1->getParameterID());
  // printf("%d \n", p2->getParameterID());
  // printf("%d \n", p3->getParameterID());

  //void (AbstractParameter::*)(int)’ to ‘void (*)(int)’

  /*
  void (*fooint)(int) = 0;
  printf("%p %p \n", fooint, hello);
  fooint = hello; 
  printf("%p %p \n", fooint, hello);

  void (*foodouble)(double) = 0;
  foodouble = hello2; 
  printf("%p %p\n", foodouble, hello2);

  void (*foos)(AbstractParameter*,int) = 0;
  foos = setParameterID;
  foos(p1,5);
  foos(p2,32);
  foos(p3,0);  

  int (*foop)(AbstractParameter*) = 0;
  foop = getParameterID;
  printf("%p %p\n", foop, getParameterID);
  printf("%d \n", foop(p1));
  printf("%d \n", foop(p2));
  printf("%d \n", foop(p3));

  */

  //  void (*set)(Object*, double) = 0;
  // set = &Mass::setMass;
  //set = &Spring::setStiffness;

  // double (*get)(Object*) = 0;
  // get = &Mass::getMass;
  // get = &Spring::getStiffness;

  // Mass mass;
  // Spring spring;

  // //update((Object) &mass, 1.0);
  // set((Object*)&mass, 1.0);
  // Mass::setMass((Object*)&mass, 1.1);
  // printf("%f \n", get((Object*)&mass));

  // set((Object*)&mass, 2.0);
  // printf("%f \n", get((Object*)&mass));

  // std::map<int,void(*)(Object*,double)> fmap;
  // fmap.insert(pair<int,void(*)(Object*,double)>(0,Mass::setMass));

  //void (Object::*update)(Object*,double) = 0;
  // update = Mass::setMass;
  // update(&mass, 1.0);

  //  typedef int(AbstractParameter::*MemberFunction)();  // Please do this!
  //  MemberFunction foop = &p1->getParameterID;

  // std::map<int,void(*)(int)> fmap;
  // fmap.insert(pair<int,void(*)(int)>(p1->getParameterID(), p1->setParameterID));

  //  fmap.insert(pair<int,void(*)(int)>(p2->getParameterID(), hello));

  // fmap[1]->setParameter(2);

  //  fmap.insert(pair<int,void(*)(double)>(p3->getParameterID(), hello2));
  //  fmap.insert(pair<int,void(*)(double)>(p4->getParameterID(), hello2));

  // 
  //setParameter(1);
  //  setParameter(2);

  int pmax[] = {3,3,4,4,2};
  int nqpts[] = {4,4,5,5,3};
  
  printf("max orders = ");
  for (int i = 0; i < nvars; i++){
    printf("%d ", pmax[i]);
  }
  printf("\nquadrature points = ");
  for (int i = 0; i < nvars; i++){
    printf("%d ", nqpts[i]);
  }
  printf("\n");

  // Initialize basis
  pc->initializeBasis(pmax);
  int nbasis = pc->getNumBasisTerms();

  // Initialize quadrature
  pc->initializeQuadrature(nqpts);  
  int nqpoints = pc->getNumQuadraturePoints();
  
  // int degs[nvars];
  // for (int k = 0; k < nbasis; k++){
  //   printf("\n");
  //   pc->getBasisParamDeg(k, degs);  
  //   for (int i = 0; i < nvars; i++){
  //     printf("%d ", degs[i]);
  //   }
  // }

  printf("%d %d \n", nqpoints, nbasis);
  
  // Space for quadrature points and weights
  double *zq = new double[nvars];
  double *yq = new double[nvars];
  double wq;

  // for (int q = 0; q < nqpoints; q++){
  //   pc->quadrature(q, zq, yq, &wq);
  // }

  for (int k = 0; k < nbasis; k++){
    for (int q = 0; q < nqpoints; q++){
      pc->quadrature(q, zq, yq, &wq);
      pc->basis(k,zq);
      // printf("%6d %6d %13.6f\n", k, q, pc->basis(k,zq));
    }
  }
    
  return 1;
}
