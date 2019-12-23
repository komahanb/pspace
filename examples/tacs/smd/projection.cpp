#include"smd.h"

#include "TACSCreator.h"
#include "TACSAssembler.h"
#include "TACSIntegrator.h"

#include "TACSFunction.h"

#include "TACSPotentialEnergy.h"
#include "TACSDisplacement.h"
#include "TACSKSFunction.h"

#include "ParameterContainer.h"
#include "ParameterFactory.h"

#include "TACSStochasticElement.h"

#include "TACSStochasticFunction.h"
#include "TACSKSStochasticFunction.h"

#include "TACSStochasticFMeanFunction.h"
#include "TACSStochasticFFMeanFunction.h"

void updateElement( TACSElement *elem, TacsScalar *vals ){
  SMD *smd = dynamic_cast<SMD*>(elem);
  if (smd != NULL) {
    smd->c = vals[0];
    // printf("smd parameters are %.17e %.17e %.17e \n", smd->m, smd->c, smd->k);
  } else {
    printf("Element mismatch while updating...");
  }
}

int main( int argc, char *argv[] ){

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank; 
  MPI_Comm_rank(comm, &rank); 

  FILE *fp = fopen("projection-smd-ks.dat", "w");

  for (int ii = 0; ii < 10; ii++){
  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *c = factory->createNormalParameter(0.2, 0.1, ii);

  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(c);
  pc->initialize();

  int nsterms = pc->getNumBasisTerms();

  TacsScalar mass = 2.5 + 1.0e-30j;
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
  const int num_funcs = 4;
  const int ks = 1;
  double ksweight = 10000.0;
    
  TACSFunction *pe, *disp;
  if (!ks){
    // Deterministic Integral
    pe = new TACSPotentialEnergy(tacs);
    disp = new TACSDisplacement(tacs);
  } else {
    // Deterministic KS
    pe = new TACSKSFunction(tacs, TACS_POTENTIAL_ENERGY_FUNCTION, ksweight);
    disp = new TACSKSFunction(tacs, TACS_DISPLACEMENT_FUNCTION, ksweight);
  }

  TACSFunction *spe, *sdisp;
  TACSFunction *spe2, *sdisp2;
  if (!ks){

    // Stochastic Integral
    spe = new TACSStochasticFMeanFunction(tacs, pe, pc, TACS_POTENTIAL_ENERGY_FUNCTION, FUNCTION_MEAN);
    sdisp = new TACSStochasticFMeanFunction(tacs, disp, pc, TACS_DISPLACEMENT_FUNCTION, FUNCTION_MEAN);

    spe2 = new TACSStochasticFFMeanFunction(tacs, pe, pc, TACS_POTENTIAL_ENERGY_FUNCTION, FUNCTION_VARIANCE);
    sdisp2 = new TACSStochasticFFMeanFunction(tacs, disp, pc, TACS_DISPLACEMENT_FUNCTION, FUNCTION_VARIANCE);

  } else {
    
    // Stochastic KS
    spe = new TACSKSStochasticFunction(tacs, pe, pc, TACS_POTENTIAL_ENERGY_FUNCTION, FUNCTION_MEAN, ksweight);
    sdisp = new TACSKSStochasticFunction(tacs, disp, pc, TACS_DISPLACEMENT_FUNCTION, FUNCTION_MEAN, ksweight);

    spe2 = new TACSKSStochasticFunction(tacs, pe, pc, TACS_POTENTIAL_ENERGY_FUNCTION, FUNCTION_VARIANCE, ksweight);
    sdisp2 = new TACSKSStochasticFunction(tacs, disp, pc, TACS_DISPLACEMENT_FUNCTION, FUNCTION_VARIANCE, ksweight);

  }

  TACSFunction **funcs = new TACSFunction*[num_funcs];
  funcs[0] = spe;
  funcs[1] = spe2;  
  funcs[2] = sdisp;
  funcs[3] = sdisp2;    

  TACSBVec *dfdx1 = tacs->createDesignVec();
  TACSBVec *dfdx2 = tacs->createDesignVec();
  TACSBVec *dfdx3 = tacs->createDesignVec();  
  TACSBVec *dfdx4 = tacs->createDesignVec();

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

  // Compute mean
  TacsScalar pemean, pe2mean, pevar;
  TacsScalar umean , u2mean, uvar;
  pemean  = ftmp[0];
  pe2mean = ftmp[1];
  umean   = ftmp[2];
  u2mean  = ftmp[3];

  pevar = pe2mean - pemean*pemean;
  uvar  = u2mean - umean*umean;
  printf("Expectations : %.17e %.17e\n", RealPart(pemean), RealPart(umean));
  printf("Variance     : %.17e %.17e\n", RealPart(pevar), RealPart(uvar));

  // if (!ks){
  //   TACSStochasticVarianceFunction *sspe, *ssdisp;
  //   sspe = dynamic_cast<TACSStochasticFMeanFunction*>(spe);
  //   ssdisp = dynamic_cast<TACSStochasticFMeanFunction*>(sdisp);
  //   printf("potential energy E = %.17e V = %.17e \n", sspe->getExpectation(), sspe->getVariance());
  //   printf("displacement     E = %.17e V = %.17e \n", ssdisp->getExpectation(), ssdisp->getVariance());

  //   sspe2 = dynamic_cast<TACSStochasticFFMeanFunction*>(spe);
  //   ssdisp2 = dynamic_cast<TACSStochasticFFMeanFunction*>(sdisp);
  //   printf("potential energy E = %.17e V = %.17e \n", sspe->getExpectation(), sspe->getVariance());
  //   printf("displacement     E = %.17e V = %.17e \n", ssdisp->getExpectation(), ssdisp->getVariance());


  // } else {
  //   TACSKSStochasticFunction *sspe, *ssdisp;
  //   sspe = dynamic_cast<TACSKSStochasticFunction*>(spe);
  //   ssdisp = dynamic_cast<TACSKSStochasticFunction*>(sdisp);
  //   printf("ks potential energy E = %.17e V = %.17e \n", sspe->getExpectation(), sspe->getVariance());
  //   printf("ks displacement     E = %.17e V = %.17e \n", ssdisp->getExpectation(), ssdisp->getVariance());
  // }

  bdf->getGradient(0, &dfdx1);
  bdf->getGradient(1, &dfdx2);
  bdf->getGradient(2, &dfdx3);
  bdf->getGradient(3, &dfdx4);

  TacsScalar *pemeanderiv, *pe2meanderiv, *umeanderiv, *u2meanderiv;
  dfdx1->getArray(&pemeanderiv);
  dfdx2->getArray(&pe2meanderiv);
  dfdx3->getArray(&umeanderiv);
  dfdx4->getArray(&u2meanderiv);

  //  printf("dE{ u  }/dx = %.17e %.17e \n", RealPart(umeanderiv[0]), RealPart(umeanderiv[1]));
  //  printf("dE{ pe }/dx = %.17e %.17e \n", RealPart(pemeanderiv[0]), RealPart(pemeanderiv[1]));

  // Find the derivative of variance
  dfdx2->axpy(-2.0*pemean, dfdx1);
  dfdx4->axpy(-2.0*umean, dfdx3);  

  //  printf("dV{ u  }/dx = %.17e %.17e \n", RealPart(u2meanderiv[0]), RealPart(u2meanderiv[1]));
  //  printf("dV{ pe }/dx = %.17e %.17e \n", RealPart(pe2meanderiv[0]), RealPart(pe2meanderiv[1]));

  double dh = 1.0e-30;
  printf("CS dE{ u  }/dx = %.17e %.17e %.17e \n", RealPart(umeanderiv[0]), ImagPart(umean)/dh,
         RealPart(umeanderiv[0]) - ImagPart(umean)/dh);
  printf("CS dE{ pe }/dx = %.17e %.17e %.17e \n", RealPart(pemeanderiv[0]), ImagPart(pemean)/dh,
         RealPart(pemeanderiv[0]) - ImagPart(pemean)/dh);
  printf("CS dV{ u  }/dx = %.17e %.17e %.17e \n", RealPart(u2meanderiv[0]), ImagPart(uvar)/dh,
         RealPart(u2meanderiv[0]) - ImagPart(uvar)/dh);
  printf("CS dV{ pe }/dx = %.17e %.17e %.17e \n", RealPart(pe2meanderiv[0]), ImagPart(pevar)/dh,
         RealPart(pe2meanderiv[0]) - ImagPart(pevar)/dh);


  fprintf(fp, "%d  %.17e  %.17e  %.17e  %.17e  %.17e  %.17e  %.17e  %.17e\n", 
          ii+1,
          RealPart(pemean),
          RealPart(pevar),
          RealPart(pemeanderiv[0]),
          RealPart(pe2meanderiv[0]),
          ImagPart(pemean)/dh,
          ImagPart(pevar)/dh,    
          RealPart(pemeanderiv[0]) - ImagPart(pemean)/dh,
          RealPart(pe2meanderiv[0]) - ImagPart(pevar)/dh);
    }

    fclose(fp);

  MPI_Finalize();  
  return 0;
}
