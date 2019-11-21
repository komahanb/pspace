#include"smd.h"

#include "TACSCreator.h"
#include "TACSAssembler.h"
#include "TACSIntegrator.h"

#include "TACSFunction.h"
#include "TACSPotentialEnergy.h"
#include "TACSDisplacement.h"

int main( int argc, char *argv[] ){
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank; 
  MPI_Comm_rank(comm, &rank); 

  TacsScalar mass = 2.5;
  TacsScalar damping = 0.2;
  TacsScalar stiffness = 5.0;
  TACSElement *smd = new SMD(mass, damping, stiffness); 
  smd->incref();

  // Assembler information to create TACS  
  int nelems = 1;
  int nnodes = 1;  
  int vars_per_node = 1;

  // Array of elements
  TACSElement **elems = new TACSElement*[nelems];
  elems[0] = smd;

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

  //---------------------------------------------------------------//  
  // Setup function evaluation within TACS
  //---------------------------------------------------------------//
  
  const int num_funcs = 2;
  TACSFunction *pe, *disp;
  pe    = new TACSPotentialEnergy(tacs); 
  disp  = new TACSDisplacement(tacs); 

  TACSFunction **funcs = new TACSFunction*[num_funcs];
  funcs[0] = pe;
  funcs[1] = disp;    

  TacsScalar *fvals = new TacsScalar[ num_funcs ];
  memset(fvals, 0, num_funcs*sizeof(TacsScalar));  

  TACSBVec *dfdx1 = tacs->createDesignVec();  dfdx1->incref();
  TACSBVec *dfdx2 = tacs->createDesignVec();  dfdx2->incref();
  
  //---------------------------------------------------------------//
  // Create the integrator class
  //---------------------------------------------------------------//

  double tinit = 0.0;
  double tfinal = 10.0;
  int nsteps = 100;
  int time_order = 2;     
  TACSIntegrator *bdf = new TACSBDFIntegrator(tacs, tinit, tfinal, nsteps, time_order);
  bdf->incref();
  bdf->setAbsTol(1e-12);
  bdf->setPrintLevel(2);
  bdf->setFunctions(num_funcs, funcs);
  bdf->integrate();  
  bdf->integrateAdjoint();
  
  bdf->evalFunctions(fvals);
  printf("pe = %e, u = %e \n", fvals[0], fvals[1]);
  bdf->getGradient(0, &dfdx1);
  bdf->getGradient(1, &dfdx2);
  
  TacsScalar *dfdx1vals;
  dfdx1->getArray(&dfdx1vals);
  printf("d{pe}dm = %e %e \n", dfdx1vals[0], dfdx1vals[1]);

  TacsScalar *dfdx2vals;
  dfdx2->getArray(&dfdx2vals);
  printf("d{u}dk  = %e %e \n", dfdx2vals[0], dfdx2vals[1]);

  // clear allocated heap
  delete [] funcs;
  dfdx1->decref();
  dfdx2->decref();  
  smd->decref();
  MPI_Finalize();
  
  return 0;
}
