#include "TACSAssembler.h"
#include "TACSIntegrator.h"
#include "ParameterContainer.h"
#include "ParameterFactory.h"
#include "TACSStochasticElement.h"
#include "TACSCreator.h"
#include "ppa.h"

void updatePPA( TACSElement *elem, TacsScalar *vals ){
  PPA *ppa = dynamic_cast<PPA*>(elem);
  if (ppa != NULL) {
    ppa->m  = vals[0];
    ppa->If = vals[1];
  }
}

PPA::PPA( TacsScalar xcm, TacsScalar xf, 
            TacsScalar m  , TacsScalar If, 
            TacsScalar ch , TacsScalar ca,
            TacsScalar kh , TacsScalar ka ){
  this->xcm = xcm;
  this->xf  = xf;
  this->m   = m;
  this->If  = If;
  this->ch  = ch;
  this->ca  = ca;
  this->kh  = kh;  
  this->ka  = ka;  
  this->s   = m*(xcm-xf);
}

void PPA::addResidual( int elemIndex, double time,
                         const TacsScalar X[], const TacsScalar v[],
                         const TacsScalar dv[], const TacsScalar ddv[],
                         TacsScalar res[] ){
  res[0] += m*ddv[0] +  s*ddv[1] + ch*dv[0] + kh*v[0];
  res[1] += s*ddv[0] + If*ddv[1] + ca*dv[1] + ka*v[1];
}

void PPA::getInitConditions( int elemIndex, const TacsScalar X[],
                               TacsScalar v[], TacsScalar dv[], TacsScalar ddv[] ){
  int num_vars = getNumNodes()*getVarsPerNode();
  memset(v, 0, num_vars*sizeof(TacsScalar));
  memset(dv, 0, num_vars*sizeof(TacsScalar));
  memset(ddv, 0, num_vars*sizeof(TacsScalar));

  // set init conditions
  v[0]  = 0.1;
  v[1]  = 0.2;
  dv[0] = 0.0;
  dv[1] = 0.0;
}

void PPA::addJacobian( int elemIndex, double time,
                         TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                         const TacsScalar X[], const TacsScalar v[],
                         const TacsScalar dv[], const TacsScalar ddv[],
                         TacsScalar res[], TacsScalar mat[] ){
  addResidual(elemIndex, time, X, v, dv, ddv, res);  
  mat[0] += gamma*m  + beta*ch + alpha*kh;
  mat[1] += gamma*s;
  mat[2] += gamma*s;
  mat[3] += gamma*If + beta*ca + alpha*ka;
}

int main( int argc, char *argv[] ){
  
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank; 
  MPI_Comm_rank(comm, &rank); 

  // Create TACS using PPA element
  TacsScalar xcm = 0.375,  xf = 0.25;
  TacsScalar m = 55.3291, If = 3.4581;
  TacsScalar ch = 0.0, ca = 0.0;
  TacsScalar kh = 11366.0, ka = 7002.6;
  PPA *ppa = new PPA(xcm, xf, m, If, ch, ca, kh, ka);
  ppa->incref();
 
  // Create random parameter
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *pm  = factory->createNormalParameter(55.3291, 5.3291, 3);
  AbstractParameter *pIf = factory->createUniformParameter(3.4581, 4.581, 4);
  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(pm);
  pc->addParameter(pIf);
  pc->initialize();

  int nsterms = pc->getNumBasisTerms();
  printf("nsterms = %d \n", nsterms);

  int nelems = 1;
  int nnodes = 1;  
  int vars_per_node = 2*nsterms;
 
  TACSStochasticElement *sppa = new TACSStochasticElement(ppa, pc, updatePPA);
  sppa->incref();
  ppa->decref();
 
  // Array of elements
  TACSElement **elems = new TACSElement*[nelems];
  elems[0] = sppa;

  // Node points array
  TacsScalar *X = new TacsScalar[3*nnodes];
  memset(X, 0, nnodes*sizeof(TacsScalar));

  // Connectivity array
  int *conn = new int[1];
  conn[0] = 0;

  // Connectivity pointer array
  int *ptr = new int[2];
  ptr[0] = 0;
  ptr[1] = 1;

  // Element Ids array
  int *eids = new int[nelems];
  for (int i = 0; i < nelems; i++){
    eids[i] = i;
  }

  // Creator object for TACS
  TACSCreator *creator = new TACSCreator(comm, vars_per_node);
  creator->incref();
  if (rank == 0){    
    creator->setGlobalConnectivity(nnodes, nelems, ptr, conn, eids);
    creator->setNodes(X);
  }
  creator->setElements(nelems, elems);

  TACSAssembler *tacs = creator->createTACS();
  tacs->incref();  

  creator->decref();    
  delete [] X;
  delete [] ptr;
  delete [] eids;
  delete [] conn;
  delete [] elems;

  // Create the integrator class
  TACSIntegrator *bdf = new TACSBDFIntegrator(tacs, 0.0, 1.0, 1, 2);
  bdf->incref();
  bdf->setAbsTol(1e-3);
  bdf->setRelTol(1e-3);
  bdf->setPrintLevel(2);
  bdf->integrate();

  delete pm;
  delete pIf;
  delete factory;
  delete pc;
  
  bdf->decref();
  tacs->decref();  

  MPI_Finalize();
  return 0;
}
