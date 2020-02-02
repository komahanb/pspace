#include "TACSAssembler.h"
#include "TACSIntegrator.h"
#include "ParameterContainer.h"
#include "ParameterFactory.h"
#include "TACSStochasticElement.h"
#include "TACSCreator.h"
#include "vpl.h"

void updateVPL( TACSElement *elem, TacsScalar *vals ){
  VPL *vpl = dynamic_cast<VPL*>(elem);
  if (vpl != NULL) {
    vpl->mu = vals[0];
  }
}

VPL::VPL( double mu ){
  this->mu = mu;
}

void VPL::addResidual( int elemIndex, double time,
                       const TacsScalar X[], const TacsScalar v[],
                       const TacsScalar dv[], const TacsScalar ddv[],
                       TacsScalar res[] ){
    res[0] += ddv[0] - mu*(1.0-v[0]*v[0])*dv[0] + v[0];
}

void VPL::getInitConditions( int elemIndex, const TacsScalar X[],
                             TacsScalar v[], TacsScalar dv[], TacsScalar ddv[] ){
  int num_vars = getNumNodes()*getVarsPerNode();
  memset(v, 0, num_vars*sizeof(TacsScalar));
  memset(dv, 0, num_vars*sizeof(TacsScalar));
  memset(ddv, 0, num_vars*sizeof(TacsScalar));

  // set init conditions
  v[0] = 1.0;
  dv[0] = 1.0;
}

void VPL::addJacobian( int elemIndex, double time,
                       TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                       const TacsScalar X[], const TacsScalar v[],
                       const TacsScalar dv[], const TacsScalar ddv[],
                       TacsScalar res[], TacsScalar mat[] ){
  addResidual(elemIndex, time, X, v, dv, ddv, res);
  mat[0] += gamma - beta*mu*(1.0-v[0]*v[0]) + alpha*(1.0 + 2.0*mu*v[0]*dv[0]*(1.0-v[0]*v[0]));
}

int main( int argc, char *argv[] ){
  
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank; 
  MPI_Comm_rank(comm, &rank); 

  // Create TACS using VPL element
  VPL *vpl = new VPL(1.0);
  vpl->incref();

  int nelems = 1;
  int nnodes = 1;  
  int vars_per_node = 1;

  // Array of elements
  TACSElement **elems = new TACSElement*[nelems];
  elems[0] = vpl;

  // Node points array
  TacsScalar *X = new TacsScalar[3*nnodes];
  memset(X, 0, 3*nnodes*sizeof(TacsScalar));

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

  //-----------------------------------------------------------------//
  // Create stochastic TACS
  //-----------------------------------------------------------------//
  
  // Create random parameter
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *mu = factory->createUniformParameter(0.75, 1.25, 9);
  
  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(mu);
  pc->initialize();
  
  int nsterms = pc->getNumBasisTerms();
  printf("nsterms = %d \n", nsterms);
  
  // should I copy the element instead?
  TACSStochasticElement *svpl = new TACSStochasticElement(vpl, pc, updateVPL);
  svpl->incref();
  
  TACSElement **selems = new TACSElement*[ nelems ];
  for ( int i = 0 ; i < nelems; i++ ){
    selems[i] = svpl; 
  }
  
  // Creator object for TACS
  TACSCreator *screator = new TACSCreator(comm, vars_per_node*nsterms);
  screator->incref();
  if (rank == 0){    
    screator->setGlobalConnectivity(nnodes, nelems, ptr, conn, eids);
    screator->setNodes(X);
  }
  screator->setElements(nelems, selems);

  TACSAssembler *stacs = screator->createTACS();
  stacs->incref();
  screator->decref();

  //  svpl->decref();
  //  vpl->decref(); // hold off may be

  delete [] X;
  delete [] ptr;
  delete [] eids;
  delete [] conn;
  delete [] elems;

  // Create the integrator class
  TACSIntegrator *bdf = new TACSBDFIntegrator(stacs, 0.0, 1.0, 10, 2);
  bdf->incref();
  bdf->setPrintLevel(2);
  bdf->integrate();
  bdf->writeRawSolution("vpl.dat", 1);  

  // write solution and test
  
  bdf->decref();
  tacs->decref();
  
  MPI_Finalize();
  return 0;
}
