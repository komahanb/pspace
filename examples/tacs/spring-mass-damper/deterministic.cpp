#include"smd.h"

#include "TACSCreator.h"
#include "TACSAssembler.h"
#include "TACSIntegrator.h"

#include "TACSFunction.h"
#include "TACSKSFunction.h"

#include "TACSPotentialEnergy.h"
#include "TACSDisplacement.h"

void deterministic_solve( MPI_Comm comm,
                          TacsScalar *p,
                          TacsScalar *fvals,
                          TacsScalar **dfdxvals ){
  int rank; 
  MPI_Comm_rank(comm, &rank); 

  TACSElement *smd = new SMD(p[0], p[1], p[2]); 
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
  int ks = 1;
  if (!ks) {
    pe = new TACSPotentialEnergy(tacs);
    disp = new TACSDisplacement(tacs);
  } else {
    double ksweight = 50.0;
    pe = new TACSKSFunction(tacs, TACS_POTENTIAL_ENERGY_FUNCTION, ksweight);
    disp = new TACSKSFunction(tacs, TACS_DISPLACEMENT_FUNCTION, ksweight);
  }
  disp->incref();
  pe->incref();
  
  TACSFunction **funcs = new TACSFunction*[num_funcs];
  funcs[0] = pe;
  funcs[1] = disp;    

  // KS functionals  
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
  bdf->setPrintLevel(0);
  bdf->setFunctions(num_funcs, funcs);

  bdf->integrate();  
  bdf->evalFunctions(fvals);

  bdf->integrateAdjoint();
  bdf->getGradient(0, &dfdx1);
  bdf->getGradient(1, &dfdx2);

  TacsScalar *dfdx1vals;
  TacsScalar *dfdx2vals;
  dfdx1->getArray(&dfdx1vals);
  dfdx2->getArray(&dfdx2vals);

  const int num_dvars = 2;
  dfdxvals[0][0] = dfdx1vals[0];
  dfdxvals[0][1] = dfdx1vals[1];

  dfdxvals[1][0] = dfdx2vals[0];
  dfdxvals[1][1] = dfdx2vals[1];
  
  // clear allocated heap
  delete [] X;
  delete [] ptr;
  delete [] eids;
  delete [] conn;
  delete [] elems;
  delete [] funcs;

  pe->decref();
  disp->decref();  
  dfdx1->decref();
  dfdx2->decref();  
  smd->decref();
}

int main( int argc, char *argv[] ){
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank; 
  MPI_Comm_rank(comm, &rank); 

  TacsScalar mass = 2.5;
  TacsScalar damping = 0.2;
  TacsScalar stiffness = 5.0;
  TacsScalar parameters[3] = {mass, damping, stiffness}; 

  const int num_funcs = 2;
  const int num_dvars = 2;
  TacsScalar *fvals = new TacsScalar[num_funcs];  
  TacsScalar **dfdxvals = new TacsScalar*[num_funcs];
  dfdxvals[0] = new TacsScalar[num_dvars];
  dfdxvals[1] = new TacsScalar[num_dvars];

  deterministic_solve(comm, parameters, fvals, dfdxvals);
    
  printf("pe = %.17e, u = %.17e \n", RealPart(fvals[0]), RealPart(fvals[1]));
  printf("d{pe}dm = %.17e %.17e \n", RealPart(dfdxvals[0][0]), RealPart(dfdxvals[0][1]));
  printf("d{u}dm  = %.17e %.17e \n", RealPart(dfdxvals[1][0]), RealPart(dfdxvals[1][1]));

  // Finite difference derivative check
  TacsScalar *fhvals = new TacsScalar[num_funcs]; 
  const double dh = 1.0e-10;
  
  TacsScalar dh1_parameters[3] = {mass+dh, damping, stiffness};
  deterministic_solve(comm, dh1_parameters, fhvals, dfdxvals);

  printf("df1dm %.17e \n", RealPart(fhvals[0]-fvals[0])/dh);
  printf("df2dm %.17e \n", RealPart(fhvals[1]-fvals[1])/dh);

  TacsScalar dh2_parameters[3] = {mass, damping, stiffness+dh};
  deterministic_solve(comm, dh2_parameters, fhvals, dfdxvals);

  printf("df1dk %.17e \n", RealPart(fhvals[0]-fvals[0])/dh);
  printf("df2dk %.17e \n", RealPart(fhvals[1]-fvals[1])/dh);
  
  MPI_Finalize();  
  return 0;
}
