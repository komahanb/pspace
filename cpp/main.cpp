#include <string>
#include <map>
#include <list>

#include"ParameterContainer.h"
#include"ParameterFactory.h"

class Object{
public:
  int id;
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


class SMD : public Object {
public:
  static double getMass(void* obj){
    SMD *self = (SMD*) obj;
    return self->m;
  };

  static void setMass(void* obj, double mass){
    SMD *self = (SMD*) obj;
    self->m = mass;
  };

  static double getStiffness(void* obj){
    SMD *self = (SMD*) obj;
    return self->k;
  };
  static void setStiffness(void* obj, double stiff){
    SMD *self = (SMD*) obj;
    self->k = stiff;
  };
  double k;  
  double m;
};

int main( int argc, char *argv[] ){

  Mass m1; 
  Spring s1;
  SMD smd;
  Mass *mm = new Mass;
  
  m1.id = 0;
  s1.id  = 1;
  smd.id = 2;

  //mm.id = 3;
  
  // list<void*> clist;
  // clist.push_back(&m1);
  // clist.push_back(&s1);
  // clist.push_back(&smd);
  // clist.push_back(mm);

  // list<int> elist;
  // elist.push_back(0);
  // elist.push_back(1);
  // elist.push_back(2);
  // elist.push_back(3);  

  // std::map<void*,int> cmap;
  // cmap.insert(pair<void*,int>(&m1,1));
  // printf("pointers %p %p\n", &m1, *(cmap[&m1]));


  // return 0;
  // if ( cmap[&s1]){
  //   printf("sdfsd\n");
  // }


  // use element ID instead of void pointers
  
  // std::list<void*>::iterator it;
  // for (it = elist.begin(); it != elist.end(); ++it){
  //   if (1) {
  //     // printf("%p %p %p %p\n", it, &m1, &s1, &smd);
  //     // printf("%p %p \n", it, mm);
  //   }
  // }


  // return 0;
  // Create random parameterse
  ParameterFactory *factory = new ParameterFactory();

  AbstractParameter *p1 = factory->createNormalParameter(-4.0, 0.5, 1,
                                                         &Mass::setMass,
                                                         &Mass::getMass);
  p1->addClientID(m1.id);
  // p1->setClientFunction(&Mass::setMass);
  // p1->getClientFunction(&Mass::getMass);


  //p1->set(&m1, 2.231324);
  //printf("%e\n", p1->get(&m1));

  // p1->updateValue(8700.0);
  // printf("%f\n", p1->getValue());

  // p1->elem = m1;
  // p1->set = &Mass::setMass;
  // p1->get = &Mass::getMass;
  // p1->set(p1->elem, 2.231324);

  
  AbstractParameter *p2 = factory->createUniformParameter(-5.0, 4.0, 1,
                                                          &SMD::setMass,
                                                          &SMD::getMass);  
  p2->addClientID(smd.id);
    
  // p2->setClientFunction(&Spring::setStiffness);
  // p2->getClientFunction(&Spring::getStiffness);

  // p2->updateValue(3.0);
  // printf("%f\n", p2->getValue());

  // p2->updateValue(4.0);
  // printf("%f\n", p2->getValue());
  
  AbstractParameter *p3 = factory->createExponentialParameter(6.0, 1.0, 1,
                                                              &SMD::setStiffness,
                                                              &SMD::getStiffness);
  p3->addClientID(smd.id);

  // maybe this?
  // p3->addClient(smd.id, SMD::setStiffness);

  // or use one pc for each element
  
  //AbstractParameter *p4 = factory->createExponentialParameter(6.0, 1.0);
  // AbstractParameter *p5 = factory->createNormalParameter(-4.0, 0.5);  

  // Create container and add random paramters
  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(p1);
  pc->addParameter(p2);
  pc->addParameter(p3);
  // pc->addParameter(p4);
  // pc->addParameter(p5);


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

  // int pmax[] = {3,3,4,4,2};
  // int nqpts[] = {4,4,5,5,3};
  
  // printf("max orders = ");
  // for (int i = 0; i < nvars; i++){
  //   printf("%d ", pmax[i]);
  // }
  // printf("\nquadrature points = ");
  // for (int i = 0; i < nvars; i++){
  //   printf("%d ", nqpts[i]);
  // }
  // printf("\n");
  
  pc->initialize();

  // Initialize basis
  //  pc->initializeBasis(pmax);
  int nbasis = pc->getNumBasisTerms();

  // Initialize quadrature
  // pc->initializeQuadrature(nqpts);  
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
  

  // for (int k = 0; k < nbasis; k++){
  //   for (int q = 0; q < nqpoints; q++){
  //     pc->quadrature(q, zq, yq, &wq);
  //     // pc->updateParameters(&m1, yq);
  //     // pc->basis(k,zq);
  //     // printf("%6d %6d %13.6f\n", k, q, pc->basis(k,zq));
  //   }
  // }

  
  for (int k = 0; k < nbasis; k++){
    for (int q = 0; q < nqpoints; q++){
      pc->quadrature(q, zq, yq, &wq);
      pc->updateParameters(m1.id, &m1, yq);
      printf("reg mass = %f %f \n", yq[0], m1.m);
      //      pc->updateParameters(&smd, yq);
      printf("smd mass = %f %f \n", yq[1], smd.m );
      printf("smd stif = %f %f \n", yq[2], smd.k );
      // pc->basis(k,zq);
      // printf("%6d %6d %13.6f\n", k, q, pc->basis(k,zq));
    }
  }

  return 1;
}
