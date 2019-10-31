#include "TACSAssembler.h"
#include "TACSIntegrator.h"
#include "ParameterContainer.h"
#include "ParameterFactory.h"
#include "TACSStochasticElement.h"
#include "TACSCreator.h"
#include "smd.h"

SMD :: SMD(double m, double c, double k){
  this->m = m;
  this->c = c;
  this->k = k;  
}

void SMD :: addResidual( double time, TacsScalar res[],
                         const TacsScalar Xpts[],
                         const TacsScalar vars[],
                         const TacsScalar dvars[],
                         const TacsScalar ddvars[] ){
  res[0] += m*ddvars[0] + c*dvars[0] + k*vars[0];
}

void SMD :: getInitConditions( TacsScalar vars[],
                               TacsScalar dvars[],
                               TacsScalar ddvars[],
                               const TacsScalar Xpts[] ){
  // zero values
  memset(vars, 0, numVariables()*sizeof(TacsScalar));
  memset(dvars, 0, numVariables()*sizeof(TacsScalar));
  memset(ddvars, 0, numVariables()*sizeof(TacsScalar));

  // set init conditions
  vars[0] = 1.0;
  dvars[0] = -1.0;
}

TACSAssembler *createTACS(){  
}

int main( int argc, char *argv[] ){
  
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank; 
  MPI_Comm_rank(comm, &rank); 

  // Create TACS using SMD element
  SMD *smd = new SMD(1.0, 0.1, 5.0);
  smd->incref();
  
  int nelems = 1;
  int nnodes = 1;  
  int vars_per_node = 1;

  // Array of elements
  TACSElement **elems = new TACSElement*[nelems];
  elems[0] = smd;

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
  creator->setElements(elems, nelems);

  TACSAssembler *tacs = creator->createTACS();
  tacs->incref();  

  creator->decref();
  smd->decref();
  delete [] X;
  delete [] ptr;
  delete [] eids;
  delete [] conn;
  delete [] elems;

  // Create the integrator class
  TACSIntegrator *bdf = new TACSBDFIntegrator(tacs, 0.0, 1.0, 10, 2);
  bdf->incref();
  bdf->setAbsTol(1e-3);
  bdf->setRelTol(1e-3);
  bdf->setPrintLevel(2);
  bdf->integrate();
  
  bdf->decref();
  tacs->decref();
  
  MPI_Finalize();
  return 0;
}
