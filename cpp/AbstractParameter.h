#ifndef ABSTRACT_PARAMETER
#define ABSTRACT_PARAMETER

#include "GaussianQuadrature.h"
#include "OrthogonalPolynomials.h"
#include <stdio.h>
#include <list>
#include <map>

class AbstractParameter {
 public:
  // Constructor and destructor
  AbstractParameter();
  ~AbstractParameter();

  // Deferred procedures
  virtual void quadrature(int npoints, double *z, double *y, double *w) = 0;
  virtual double basis(double z, int d) = 0;

  // Implemented procedures
  int getParameterID();  
  void setParameterID(int pid);
  void setMaxDegree(int dmax){
    this->dmax = dmax;
  }
  int getMaxDegree(){
    return this->dmax;
  }
  
  // Function pointer to set values
  void setClientFunction( void (*func)(void*, double) ){
    this->set = func;
  }

  // Function pointer to retrieve values
  void getClientFunction( double (*func)(void*) ){
    this->get = func;
  }

  void (*set)(void*, double); // handle to set things
  double (*get)(void*);   // handle to get things

  //  void *elem; // points to element of consitituvive object to update 

 
  int match(int cid){
    std::list<int>::iterator it;
    for (it = clist.begin(); it != clist.end(); ++it){
      if (cid == *it){
        return 1;
      }
    }
    return 0;
  }
  
  // Make this a part of TACS maybe?
  // Fancy test stuff
  void addClient(int cid, void (*fset)(void*, double)){ 
    //    this->elem = obj;
    //    cmap.insert(std::pair<int, ClientData// void(*)(void*,double)>(cid,fset));
  }
    
  // Make this a part of TACS maybe?
  // Fancy test stuff
  void addClientID(int cid){ 
    //    this->elem = obj;
    this->clist.push_back(cid);
    //    cmap.insert(std::pair<void*,int>(obj,0));
  }

  void updateValue(int cid, void *obj, double value){
    if (!obj){
      printf("NULL Object\n");
    }
    if ( match(cid) ){
      this->set(obj, value);
    }
  };
  
  double getValue(int cid, void *obj){ 
    if ( match(cid) ){
      return this->get(obj); 
    } else {
      return 0.0;
    }
  }; 

 protected:
  GaussianQuadrature *gauss;
  OrthogonalPolynomials *polyn;
  
 private:
  int parameter_id;
  int dmax;

  // Maintain a list of client objects that use this parameter
  std::list<int> clist;
  std::map<int,void*> cmap;
};

#endif
