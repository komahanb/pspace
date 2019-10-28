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
  
  void setClient(void *client){ 
    this->client = client;
  }

  void updateValue(void *obj, double value){
    printf("calling cpp update \n");
    if ( this->client == obj ){
      this->set(obj, value);
    } else {
      printf("skipping update \n");
    }
  };
  
  double getValue(void *obj){ 
    printf("calling cpp get \n");
    if ( this->client == obj ){
      return this->get(obj); 
    } else {
      printf("default return \n");
      return 0.0;
    }
  }; 

  /* 
     int match(int cid){
     std::list<int>::iterator it;
     for (it = clist.begin(); it != clist.end(); ++it){
     if (cid == *it){
     return 1;
     }
     }
     return 0;
     }
     void addClientID(int cid){ 
     this->clist.push_back(cid);
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
  */

 protected:
  GaussianQuadrature *gauss;
  OrthogonalPolynomials *polyn;
  void (*set)(void*, double); // handle to set things
  double (*get)(void*);   // handle to get things  
  void *client; // points to element of consitituvive object to update 
  
 private:
  int parameter_id;
  int dmax;

  // Maintain a list of client objects that use this parameter
  std::list<int> clist;
  std::map<int,void*> cmap;
};

#endif
