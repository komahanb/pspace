#include"smd.h"

#include "TACSCreator.h"
#include "TACSAssembler.h"
#include "TACSIntegrator.h"

#include "TACSFunction.h"
#include "TACSPotentialEnergy.h"
#include "TACSDisplacement.h"

#include "ParameterContainer.h"
#include "ParameterFactory.h"

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
  pe    = new TACSPotentialEnergy(tacs); pe->incref();
  disp  = new TACSDisplacement(tacs); disp->incref();

  TACSFunction **funcs = new TACSFunction*[num_funcs];
  funcs[0] = pe;
  funcs[1] = disp;    

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
  bdf->integrateAdjoint();
  
  bdf->evalFunctions(fvals);
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

  const int num_funcs = 2;
  const int num_dvars = 2;

  int pnqpts[1] = {10};
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *c = factory->createNormalParameter(0.2, 0.1, 0);  

  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(c);
  pc->initializeQuadrature(pnqpts);

  const int nqpoints = pc->getNumQuadraturePoints();
  const int nvars = pc->getNumParameters();

  TacsScalar **f = new TacsScalar*[nqpoints];  
  for (int i = 0; i < nqpoints; i++){
    f[i] = new TacsScalar[num_funcs];  
  }

  TacsScalar ***dfdx = new TacsScalar**[nqpoints];
  for (int i = 0; i < nqpoints; i++){
    dfdx[i] = new TacsScalar*[num_funcs];
    for (int j = 0; j < num_funcs; j++){
      dfdx[i][j] = new TacsScalar[num_dvars];  
    }
  }
  
  double *zq = new double[nvars];
  double *yq = new double[nvars];
  double wq;
 
  TacsScalar mass = 2.5;
  TacsScalar damping = 0.2;
  TacsScalar stiffness = 5.0;

  for (int q = 0; q < nqpoints; q++){
    wq = pc->quadrature(q, zq, yq);
    printf("deterministic solve %d at c = %e\n", q, yq[0]);
    damping = yq[0];
    TacsScalar params[3] = {mass, damping, stiffness};
    deterministic_solve(comm, params, f[q], dfdx[q]);
    printf("\t disp = %e energy = %e \n", f[q][0], f[q][1]);
  }

  //-------------------------------------------------------------------//
  // Post processing to find statistical moments (mean and variance)
  //-------------------------------------------------------------------//

  // Function values
  TacsScalar *fmean = new TacsScalar[num_funcs];  
  memset(fmean, 0, num_funcs*sizeof(TacsScalar));
  TacsScalar *fvar = new TacsScalar[num_funcs];  
  memset(fvar, 0, num_funcs*sizeof(TacsScalar));  

  // Derivative values
  TacsScalar **dfdxmean = new TacsScalar*[num_funcs];
  for (int j = 0; j < num_funcs; j++){
    dfdxmean[j] = new TacsScalar[num_dvars];  
    memset(dfdxmean[j], 0, num_dvars*sizeof(TacsScalar));
  }
  TacsScalar **dfdxvar = new TacsScalar*[num_funcs];
  for (int j = 0; j < num_funcs; j++){
    dfdxvar[j] = new TacsScalar[num_dvars];  
    memset(dfdxvar[j], 0, num_dvars*sizeof(TacsScalar));
  }

  // Compute mean of function and derivatives
  for (int q = 0; q < nqpoints; q++){
    wq = pc->quadrature(q, zq, yq);
    for (int i = 0; i < num_funcs; i++){
      fmean[i] += wq*f[q][i];
      for (int j = 0; j < num_dvars; j++){
        dfdxmean[i][j] += wq*dfdx[q][i][j];
      }
    }
  }

  // Print output expectation
  for (int i = 0; i < num_funcs; i++){
    printf("E[f%d] = %e \n", i, fmean[i]);
  }  
  for (int i = 0; i < num_funcs; i++){
    printf("E[df%ddx] = ", i);
    for (int j = 0; j < num_dvars; j++){
      printf("%e ", dfdxmean[i][j]);
    }
    printf("\n");
  }
    
  // Compute variance of function and derivatives
  for (int q = 0; q < nqpoints; q++){
    wq = pc->quadrature(q, zq, yq);
    for (int i = 0; i < num_funcs; i++){
      fvar[i] += wq*(fmean[i]-f[q][i])*(fmean[i]-f[q][i]);      
      for (int j = 0; j < num_dvars; j++){
        dfdxvar[i][j] += wq*(dfdxmean[i][j]-dfdx[q][i][j])*(dfdxmean[i][j]-dfdx[q][i][j]);
      }
    }
  }

  // Print output variance
  for (int i = 0; i < num_funcs; i++){
    printf("V[f%d] = %e \n", i, fvar[i]);
  }  
  for (int i = 0; i < num_funcs; i++){
    printf("V[df%ddx] = ", i);
    for (int j = 0; j < num_dvars; j++){
      printf("%e ", dfdxmean[i][j]);
    }
    printf("\n");
  }

  delete [] zq;
  delete [] yq;
    
  MPI_Finalize();  
  return 0;
}
