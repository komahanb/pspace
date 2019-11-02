#include "TACSAssembler.h"
#include "TACSIntegrator.h"
#include "ParameterContainer.h"
#include "ParameterFactory.h"
#include "TACSStochasticElement.h"
#include "TACSCreator.h"
#include "smd.h"

void updateSMD( TACSElement *elem, TacsScalar *vals ){
  SMD *smd = dynamic_cast<SMD*>(elem);
  if (smd != NULL) {
    smd->m = vals[0];
    smd->c = vals[1];
    smd->k = vals[2];
    //    printf("%e %e %e \n", smd->m, smd->c, smd->k);
  } else {
    printf("Element mismatch while updating...");
  }
}

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

void SMD :: addJacobian( double time, TacsScalar J[],
                         double alpha, double beta, double gamma,
                         const TacsScalar X[],
                         const TacsScalar v[],
                         const TacsScalar dv[],
                         const TacsScalar ddv[] ){
  J[0] += gamma*m + beta*c + alpha*k;
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
  SMD *smd = new SMD(2.0, 3.0, 4.0);
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

  //-----------------------------------------------------------------//
  // Create stochastic TACS
  //-----------------------------------------------------------------//
  
  // Create random parameter
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *m = factory->createExponentialParameter(5.0, 0.5, 4);
  AbstractParameter *c = factory->createUniformParameter(0.2, 0.5, 3);
  AbstractParameter *k = factory->createNormalParameter(2.0, 3.0, 2);
 
  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(m);
  pc->addParameter(c);
  pc->addParameter(k);

  pc->initialize();
  int nsterms = pc->getNumBasisTerms();
  printf("nsterms = %d \n", nsterms);
  
  // should I copy the element instead?
  TACSStochasticElement *ssmd = new TACSStochasticElement(smd, pc);
  ssmd->incref();
  ssmd->setUpdateCallback(updateSMD);
  
  TACSElement **selems = new TACSElement*[ nelems ];
  for ( int i = 0 ; i < nelems; i++ ){
    selems[i] = ssmd; 
  }
  
  // Creator object for TACS
  TACSCreator *screator = new TACSCreator(comm, vars_per_node*nsterms);
  screator->incref();
  if (rank == 0){    
    screator->setGlobalConnectivity(nnodes, nelems, ptr, conn, eids);
    screator->setNodes(X);
  }
  screator->setElements(selems, nelems);

  TACSAssembler *stacs = screator->createTACS();
  stacs->incref();
  screator->decref();

  //  ssmd->decref();
  //  smd->decref(); // hold off may be

  delete [] X;
  delete [] ptr;
  delete [] eids;
  delete [] conn;
  delete [] elems;

  // Create the integrator class
  TACSIntegrator *bdf = new TACSBDFIntegrator(stacs, 0.0, 1.0, 10, 2);
  bdf->incref();
  bdf->setAbsTol(1e-3);
  bdf->setRelTol(1e-3);
  bdf->setPrintLevel(2);
  bdf->integrate();

  // write solution and test
  
  bdf->decref();
  tacs->decref();
  
  MPI_Finalize();
  return 0;
}

  // Test Stochastic Jacobian
  /*
  int ndof  = nsterms*vars_per_node;
  TacsScalar *v = new TacsScalar[ndof];
  TacsScalar *vdot = new TacsScalar[ndof];
  TacsScalar *vddot = new TacsScalar[ndof];
  TacsScalar *J = new TacsScalar[ndof*ndof];
  memset(J, 0, ndof*ndof*sizeof(TacsScalar));

  ssmd->addJacobian( 0.0, J,
                     0.0, 0.0, 1.0,
                     NULL,
                     v,
                     vdot,
                     vddot);
  int ctr = 0;
  for ( int i = 0 ; i < ndof; i++ ){
    for ( int j = 0 ; j < ndof; j++ ){
      printf(" %e ", J[ctr]);
      ctr ++;
    }
    printf("\n");
  }

  return 0;
  */
