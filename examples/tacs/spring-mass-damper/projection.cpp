#include"smd.h"

#include "TACSCreator.h"
#include "TACSAssembler.h"
#include "TACSIntegrator.h"

#include "TACSFunction.h"
#include "TACSPotentialEnergy.h"
#include "TACSDisplacement.h"

#include "ParameterContainer.h"
#include "ParameterFactory.h"

#include "TACSStochasticElement.h"
#include "TACSStochasticFunction.h"
#include "TACSStochasticVarianceFunction.h"

void updateElement( TACSElement *elem, TacsScalar *vals ){
  SMD *smd = dynamic_cast<SMD*>(elem);
  if (smd != NULL) {
    smd->c = vals[0];
    // printf("smd parameters are %e %e %e \n", smd->m, smd->c, smd->k);
  } else {
    printf("Element mismatch while updating...");
  }
}

int main( int argc, char *argv[] ){

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank; 
  MPI_Comm_rank(comm, &rank); 

  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *c = factory->createNormalParameter(0.2, 0.1, 5);

  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(c);
  pc->initialize();

  int nsterms = pc->getNumBasisTerms();

  TacsScalar mass = 2.5;
  TacsScalar damping = 0.2;
  TacsScalar stiffness = 5.0;
  TACSElement *smd = new SMD(mass, damping, stiffness); 
  TACSStochasticElement *ssmd = new TACSStochasticElement(smd, pc, updateElement);

  // Assembler information to create TACS  
  int nelems = 1;
  int nnodes = 1;  
  int vars_per_node = 1*nsterms;

  // Array of elements
  TACSElement **elems = new TACSElement*[nelems];
  elems[0] = ssmd;

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

  const int num_dvars = 2;
  const int num_funcs = 2;

  TACSFunction *pe, *disp;
  pe    = new TACSPotentialEnergy(tacs); 
  disp  = new TACSDisplacement(tacs); 

  TACSStochasticVarianceFunction *spe, *sdisp;
  spe   = new TACSStochasticVarianceFunction(tacs, pe, pc, TACS_POTENTIAL_ENERGY_FUNCTION, 1);
  sdisp = new TACSStochasticVarianceFunction(tacs, disp, pc, TACS_DISPLACEMENT_FUNCTION, 1);

  TACSFunction **funcs = new TACSFunction*[num_funcs];
  funcs[0] = spe;
  funcs[1] = sdisp;    

  TACSBVec *dfdx1 = tacs->createDesignVec();
  TACSBVec *dfdx2 = tacs->createDesignVec();

  TacsScalar *ftmp = new TacsScalar[ num_funcs ];
  memset(ftmp, 0, num_funcs*sizeof(TacsScalar));  

  //-----------------------------------------------------------------//
  // Create the integrator class
  //-----------------------------------------------------------------//

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
  bdf->evalFunctions(ftmp);
  bdf->integrateAdjoint();
  
  // Post processing to get moments
  for (int i = 0; i < num_funcs; i++){
    printf("projection E[f] = %e\n", ftmp[i]);
  }
  printf("potential energy E = %e V = %e \n", spe->getExpectation(), spe->getVariance());
  printf("displacement     E = %e V = %e \n", sdisp->getExpectation(), sdisp->getVariance());

  bdf->getGradient(0, &dfdx1);
  TacsScalar *dfdx1vals;
  dfdx1->getArray(&dfdx1vals);
  printf("d{pe}dm = %e %e \n", dfdx1vals[0], dfdx1vals[1]);

  bdf->getGradient(1, &dfdx2);
  TacsScalar *dfdx2vals;
  dfdx2->getArray(&dfdx2vals);
  printf("d{u}dk  = %e %e \n", dfdx2vals[0], dfdx2vals[1]);



  printf("[c] Get mean derivative \n");
  printf("[ ] Get variance \n");
  printf("[ ] Get variance derivative \n");
  printf("[ ] ks implementation \n");
  MPI_Finalize();  
  return 0;
}
