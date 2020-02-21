#include "TACSCreator.h"
#include "TACSAssembler.h"
#include "TACSIntegrator.h"
#include "TACSRigidBody.h"
#include "TACSKinematicConstraints.h"
#include "SquareSection.h"
#include "MITC3.h"
#include "TACSKSFailure.h"
//#include "TACSKSDisplacement.h"
#include "TACSStructuralMass.h"
#include "TACSConstitutiveVerification.h"
#include "TACSElementVerification.h"

#include "ParameterContainer.h"
#include "ParameterFactory.h"

#include "TACSStochasticElement.h"
#include "TACSKSStochasticFunction.h"
#include "TACSStochasticFunction.h"

void getDeterministicStates( ParameterContainer *pc, 
                             TACSElement *delem,
                             TACSElement *selem, 
                             const TacsScalar v[],
                             const TacsScalar dv[],
                             const TacsScalar ddv[], 
                             TacsScalar *zq,
                             TacsScalar *uq,
                             TacsScalar *udq,
                             TacsScalar *uddq
                             ){
  int ndvpn   = delem->getVarsPerNode();
  int nsvpn   = selem->getVarsPerNode();
  int nddof   = delem->getNumVariables();
  int nsdof   = selem->getNumVariables();
  int nsterms = pc->getNumBasisTerms();
  int nnodes  = selem->getNumNodes();

  memset(uq  , 0, nddof*sizeof(TacsScalar));
  memset(udq , 0, nddof*sizeof(TacsScalar));
  memset(uddq, 0, nddof*sizeof(TacsScalar));

  // Evaluate the basis at quadrature node and form the state
  // vectors
  for (int n = 0; n < nnodes; n++){
    for (int k = 0; k < nsterms; k++){
      TacsScalar psikz = pc->basis(k,zq);
      int lptr = n*ndvpn;
      int gptr = n*nsvpn + k*ndvpn;
      for (int d = 0; d < ndvpn; d++){        
        uq[lptr+d] += v[gptr+d]*psikz;
        udq[lptr+d] += dv[gptr+d]*psikz;
        uddq[lptr+d] += ddv[gptr+d]*psikz;
      }
    }
  }
} 

void updateBeam1( TACSElement *elem, TacsScalar *vals ){
  MITC3 *mitc3 = dynamic_cast<MITC3*>(elem);
  if (mitc3 != NULL) {
    TACSTimoshenkoConstitutive *stiff = dynamic_cast<TACSTimoshenkoConstitutive*>(mitc3->getConstitutive());
    if (stiff){
      stiff->incref();      TacsScalar rho[4];
      stiff->getProperties(rho, NULL, NULL);
      rho[0] = vals[0];
      stiff->setProperties(rho, NULL, NULL);
      stiff->decref();
    }
  } else {
    printf("Element mismatch while updating...");
  }
}

void updateBeam2( TACSElement *elem, TacsScalar *vals ){
  MITC3 *mitc3 = dynamic_cast<MITC3*>(elem);
  if (mitc3 != NULL) {
    TACSTimoshenkoConstitutive *stiff = dynamic_cast<TACSTimoshenkoConstitutive*>(mitc3->getConstitutive());
    if (stiff){
      stiff->incref();      TacsScalar rho[4];
      stiff->getProperties(rho, NULL, NULL);
      rho[0] = vals[1];
      stiff->setProperties(rho, NULL, NULL);
      stiff->decref();
    }
  } else {
    printf("Element mismatch while updating...");
  }
}

void updateRevoluteDriver( TACSElement *elem, TacsScalar *vals ){
  TACSRevoluteDriver *revDriverA = dynamic_cast<TACSRevoluteDriver*>(elem);
  if (revDriverA != NULL) {
    revDriverA->setSpeed(vals[0]);
    // printf("updating driver with speed %e \n...", vals[0]);
  } else {
    printf("Element mismatch while updating...");
  }
}

void updateRevoluteConstraint( TACSElement *elem, TacsScalar *vals ){
  TACSRevoluteConstraint *revConstraint = dynamic_cast<TACSRevoluteConstraint*>(elem);
  if (revConstraint != NULL) {
    TacsScalar theta = (vals[0]*M_PI/180.0);
    TACSGibbsVector *revDir = new TACSGibbsVector(sin(theta), 0.0, cos(theta));
    revConstraint->setRevoluteAxis(revDir);    
  } else {
    printf("Element mismatch while updating...");
  }
}

/*
  Create and return the TACSAssembler object for the four bar
  mechanism as described by Bachau

  B ------------------- C
  |                     |
  |                     |
  |                     |
  A                     D

  Length between A and B = 0.12 m
  Length between B and C = 0.24 m
  Length between C and D = 0.12 m

  A, B and D are revolute joints in the plane perpendicular to the
  plane of the mechanism

  C is a revolute joint in a plane +5 degrees along the DC axis of the
  beam

  Beam properties:

  Young's modulus 207 GPa, nu = 0.3

  Bars 1 and 2 are square and of dimension 16 x 16 mm
  Bar 3 is square and of dimension 8 x 8 mm
*/
TACSAssembler *four_bar_mechanism( int nA, int nB, int nC, ParameterContainer *pc){
  int rank; 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

  // Set the gravity vector
  TACSGibbsVector *gravity = new TACSGibbsVector(0.0, 0.0, -9.81);

  // Set the points b, c and d
  TACSGibbsVector *ptB = new TACSGibbsVector(0.0, 0.12, 0.0);
  TACSGibbsVector *ptC = new TACSGibbsVector(0.24, 0.12, 0.0);
  TACSGibbsVector *ptD = new TACSGibbsVector(0.24, 0.0, 0.0);

  // Create the revolute direction for B and D
  TACSGibbsVector *revDirA = new TACSGibbsVector(0.0, 0.0, 1.0);
  TACSGibbsVector *revDirB = new TACSGibbsVector(0.0, 0.0, 1.0);
  TACSGibbsVector *revDirD = new TACSGibbsVector(0.0, 0.0, 1.0);

  // Create the revolute direction for C
  TacsScalar theta = (5.0*M_PI/180.0);
  TACSGibbsVector *revDirC = new TACSGibbsVector(sin(theta), 0.0, cos(theta));

  // Create the revolute constraints
  TacsScalar omega = -0.6; // rad/seconds
  int fixed_point = 1;
  int not_fixed = 0;

  TACSRevoluteDriver *revDriverA =
    new TACSRevoluteDriver(revDirA, omega);
  TACSRevoluteConstraint *revB =
    new TACSRevoluteConstraint(not_fixed, ptB, revDirB);
  TACSRevoluteConstraint *revC =
    new TACSRevoluteConstraint(not_fixed, ptC, revDirC);
  TACSRevoluteConstraint *revD =
    new TACSRevoluteConstraint(fixed_point, ptD, revDirD);

  // Set the reference axes for each beam
  TacsScalar axis_A[] = {-1.0, 0.0, 0.0};
  TacsScalar axis_B[] = {0.0, 1.0, 0.0};
  TacsScalar axis_C[] = {1.0, 0.0, 0.0};
  
  int nsterms = pc->getNumBasisTerms();
  printf("nsterms = %d \n", nsterms);

  // Set the material properties
  TacsScalar density = 7800.0;
  TacsScalar E = 207e9;
  TacsScalar nu = 0.3;
  TacsScalar G = 0.5*E/(1.0 + nu);

  //TacsScalar wA = 5.624455e-03;
  //TacsScalar wB = 1.181739e-02;
  TacsScalar wA = 0.016;// + 1.0e-30j;
  TacsScalar wB = 0.008;
  int wANum = 0, wBNum = 1;

  TACSTimoshenkoConstitutive *stiffA =
    new SquareSection(density, E, G, wA, wANum, axis_A, 0.12);

  TACSTimoshenkoConstitutive *stiffB =
    new SquareSection(density, E, G, wA, wANum, axis_B, 0.24);

  TACSTimoshenkoConstitutive *stiffC =
    new SquareSection(density, E, G, wB, wBNum, axis_C, 0.12);

  // Set up the connectivity
  MITC3 *beamA = new MITC3(stiffA, gravity);
  MITC3 *beamB = new MITC3(stiffB, gravity);
  MITC3 *beamC = new MITC3(stiffC, gravity);

  // Create stochastic elements
  TACSStochasticElement *sbeamA      = new TACSStochasticElement(beamA, pc, NULL);
  TACSStochasticElement *sbeamB      = new TACSStochasticElement(beamB, pc, NULL);
  TACSStochasticElement *sbeamC      = new TACSStochasticElement(beamC, pc, NULL);
  TACSStochasticElement *srevDriverA = new TACSStochasticElement(revDriverA, pc, NULL);
  TACSStochasticElement *srevB       = new TACSStochasticElement(revB, pc, NULL);
  TACSStochasticElement *srevC       = new TACSStochasticElement(revC, pc, updateRevoluteConstraint);
  TACSStochasticElement *srevD       = new TACSStochasticElement(revD, pc, NULL);

  // Set the number of nodes in the mesh
  int nnodes = (2*nA+1) + (2*nB+1) + (2*nC+1) + 4;

  // Set the number of elements
  int nelems = nA + nB + nC + 4;

  // Create the connectivities
  TacsScalar *X = new TacsScalar[ 3*nnodes ];
  memset(X, 0, 3*nnodes*sizeof(TacsScalar));

  int *ptr = new int[ nelems+1 ];
  int *conn = new int[ 3*nelems ];
  TACSElement **elems = new TACSElement*[ nelems ];

  // Set the nodes numbers and locations
  int *nodesA = new int[ 2*nA+1 ];
  int *nodesB = new int[ 2*nB+1 ];
  int *nodesC = new int[ 2*nC+1 ];
  int n = 0;
  for ( int i = 0; i < 2*nA+1; i++, n++ ){
    nodesA[i] = n;
    X[3*n+1] = 0.12*i/(2*nA);
  }
  for ( int i = 0; i < 2*nB+1; i++, n++ ){
    nodesB[i] = n;
    X[3*n] = 0.24*i/(2*nB);
    X[3*n+1] = 0.12;
  }
  for ( int i = 0; i < 2*nC+1; i++, n++ ){
    nodesC[i] = n;
    X[3*n] = 0.24;
    X[3*n+1] = 0.12*(1.0 - 1.0*i/(2*nC));
  }

  // Set the connectivity for the beams
  int elem = 0;
  ptr[0] = 0;
  for ( int i = 0; i < nA; i++ ){
    conn[ptr[elem]] = nodesA[2*i];
    conn[ptr[elem]+1] = nodesA[2*i+1];
    conn[ptr[elem]+2] = nodesA[2*i+2];
    elems[elem] = sbeamA;
    ptr[elem+1] = ptr[elem] + 3;
    elem++;
  }

  for ( int i = 0; i < nB; i++ ){
    conn[ptr[elem]] = nodesB[2*i];
    conn[ptr[elem]+1] = nodesB[2*i+1];
    conn[ptr[elem]+2] = nodesB[2*i+2];
    elems[elem] = sbeamB;
    ptr[elem+1] = ptr[elem] + 3;
    elem++;
  }

  for ( int i = 0; i < nC; i++ ){
    conn[ptr[elem]] = nodesC[2*i];
    conn[ptr[elem]+1] = nodesC[2*i+1];
    conn[ptr[elem]+2] = nodesC[2*i+2];
    elems[elem] = sbeamC;
    ptr[elem+1] = ptr[elem] + 3;
    elem++;
  }

  // Add the connectivities for the constraints
  conn[ptr[elem]] = nodesA[0];
  conn[ptr[elem]+1] = nnodes-4;
  elems[elem] = srevDriverA;
  ptr[elem+1] = ptr[elem] + 2;
  elem++;

  conn[ptr[elem]] = nodesA[2*nA];
  conn[ptr[elem]+1] = nodesB[0];
  conn[ptr[elem]+2] = nnodes-3;
  elems[elem] = srevB;
  ptr[elem+1] = ptr[elem] + 3;
  elem++;

  conn[ptr[elem]] = nodesC[0];
  conn[ptr[elem]+1] = nodesB[2*nB];
  conn[ptr[elem]+2] = nnodes-2;
  elems[elem] = srevC;
  ptr[elem+1] = ptr[elem] + 3;
  elem++;

  conn[ptr[elem]] = nodesC[2*nC];
  conn[ptr[elem]+1] = nnodes-1;
  elems[elem] = srevD;
  ptr[elem+1] = ptr[elem] + 2;
  elem++;

  delete [] nodesA;
  delete [] nodesB;
  delete [] nodesC;

  int vars_per_node = 8*nsterms;

  // Node points array
  TacsScalar *Xpts = new TacsScalar[3*nnodes];
  memset(Xpts, 0, 3*nnodes*sizeof(TacsScalar));

  // Element Ids array
  int *eids = new int[nelems];
  for (int i = 0; i < nelems; i++){
    eids[i] = i;
  }

  // Creator object for TACS
  TACSCreator *creator = new TACSCreator(MPI_COMM_WORLD, vars_per_node);
  creator->incref();
  if (rank == 0){    
    creator->setGlobalConnectivity(nnodes, nelems, ptr, conn, eids);
    creator->setNodes(Xpts);
  }
  creator->setElements(nelems, elems);

  //  TACSAssembler *assembler = creator->createTACS();
  //assembler->incref();  
  //creator->decref(); 
  
  // Create the TACSAssembler object

  TACSAssembler *assembler = new TACSAssembler(MPI_COMM_WORLD, 8*nsterms, nnodes, nelems);

  assembler->setElementConnectivity(ptr, conn);
  delete [] conn;
  delete [] ptr;

  assembler->setElements(elems);
  delete [] elems;

  assembler->initialize();
  
  // Set the node locations
  TACSBVec *Xvec = assembler->createNodeVec();
  Xvec->incref();
  TacsScalar *Xarray;
  Xvec->getArray(&Xarray);
  memcpy(Xarray, X, 3*nnodes*sizeof(TacsScalar));
  assembler->setNodes(Xvec);
  Xvec->decref();
  delete [] X;
  
  return assembler;
}

#ifndef OPT
int main( int argc, char *argv[] ){
  // Initialize MPI
  MPI_Init(&argc, &argv);

  ParameterFactory *factory  = new ParameterFactory();
  // AbstractParameter *pm1 = factory->createExponentialParameter(mA, 0.1, 1);
  // AbstractParameter *pm2 = factory->createExponentialParameter(mB, 0.2, 1);
  // AbstractParameter *pOmegaA = factory->createNormalParameter(-0.6, 0.06, 5);
  AbstractParameter *ptheta = factory->createNormalParameter(5.0, 2.5, 3);

  ParameterContainer *pc = new ParameterContainer();
  //pc->addParameter(pm1);
  //pc->addParameter(pm2); 
  pc->addParameter(ptheta);  
  pc->initialize();

  const int nsterms  = pc->getNumBasisTerms();
  const int nsqpts   = pc->getNumQuadraturePoints();

  // Create the finite-element model
  int nA = 2, nB = 4, nC = 2;
  TACSAssembler *assembler = four_bar_mechanism(nA, nB, nC, pc);
  assembler->incref();

  // Set the final time
  double tf = 12.0;

  // The number of total steps (100 per second)
  int num_steps = 12000;

  // Create the integrator class
  TACSIntegrator *integrator = new TACSBDFIntegrator(assembler, 0.0, tf, num_steps, 2);
  integrator->incref();

  // Set the integrator options
  integrator->setUseSchurMat(1, TACSAssembler::TACS_AMD_ORDER);
  integrator->setAbsTol(1e-7);
  integrator->setRelTol(1e-12);
  integrator->setOutputFrequency(0);
  integrator->setPrintLevel(0);
  integrator->setJacAssemblyFreq(5);
  
  // Integrate the equations of motion forward in time
  integrator->integrate();

  // Create the continuous KS function
  double ksRho = 10000.0;
  TACSKSFailure  *ksfunc = new TACSKSFailure(assembler, ksRho);
  TACSStructuralMass *fmass = new TACSStructuralMass(assembler);
  TACSKSDisplacement *ksdisp = new TACSKSDisplacement(assembler, ksRho);

  const int num_funcs = 6; // mean and variance
  TACSFunction **funcs = new TACSFunction*[num_funcs];

 TACSFunction *sfuncmass, *sffuncmass;
  sfuncmass  = new TACSStochasticFunction(assembler, fmass, pc, TACS_ELEMENT_DENSITY, FUNCTION_MEAN);
  sffuncmass = new TACSStochasticFunction(assembler, fmass, pc, TACS_ELEMENT_DENSITY, FUNCTION_VARIANCE);
  funcs[0] = sfuncmass;
  funcs[1] = sffuncmass;  

  TACSFunction *sfuncfail, *sffuncfail;
  sfuncfail  = new TACSKSStochasticFunction(assembler, ksfunc, pc, TACS_FAILURE_INDEX, FUNCTION_MEAN, ksRho);
  sffuncfail = new TACSKSStochasticFunction(assembler, ksfunc, pc, TACS_FAILURE_INDEX, FUNCTION_VARIANCE, ksRho);
  funcs[2] = sfuncfail;
  funcs[3] = sffuncfail;

  TACSFunction *sfuncdisp, *sffuncdisp;
  sfuncdisp  = new TACSKSStochasticFunction(assembler, ksfunc, pc, TACS_ELEMENT_DISPLACEMENT, FUNCTION_MEAN, ksRho);
  sffuncdisp = new TACSKSStochasticFunction(assembler, ksfunc, pc, TACS_ELEMENT_DISPLACEMENT, FUNCTION_VARIANCE, ksRho);
  funcs[4] = sfuncdisp;
  funcs[5] = sffuncdisp;

  // Create stochastic functions to set into TACS 
  integrator->setFunctions(num_funcs, funcs);

  TacsScalar fval[num_funcs];
  integrator->evalFunctions(fval);
  for ( int i = 0; i < num_funcs; i++ ){
    printf("Function values : %15.10e \n", TacsRealPart(fval[i]));
  }

  TacsScalar massmean = fval[0];
  TacsScalar massvar  = fval[1];
  TacsScalar failmean = fval[2];
  TacsScalar failvar  = fval[3];
  TacsScalar dispmean = fval[4];
  TacsScalar dispvar  = fval[5];

// #ifdef TACS_USE_COMPLEX
//   integrator->checkGradients(1e-30);
// #else
//   integrator->checkGradients(1e-6);
// #endif // TACS_USE_COMPLEX

  // Evaluate the adjoint
  integrator->integrateAdjoint();

  // Get the gradient for mass
  TACSBVec *dfdx1, *dfdx2;
  integrator->getGradient(0, &dfdx1);
  integrator->getGradient(1, &dfdx2);

  TacsScalar *massmeanderiv, *massvarderiv;
  dfdx1->getArray(&massmeanderiv);
  dfdx2->getArray(&massvarderiv);

  // Find the derivative of variance
  dfdx2->axpy(-2.0*massmean, dfdx1);

  double dh = 1.0e-30;
  printf("CS dE{ mass }/dx = %.17e %.17e %.17e \n", 
         RealPart(massmeanderiv[0]), 
         ImagPart(massmean)/dh,
         RealPart(massmeanderiv[0]) - ImagPart(massmean)/dh);
  printf("CS dV{ mass }/dx = %.17e %.17e %.17e \n", 
         RealPart(massvarderiv[0]), 
         ImagPart(massvar)/dh,
         RealPart(massvarderiv[0]) - ImagPart(massvar)/dh);

  // Get the gradient of failure
  TACSBVec *dfdx3, *dfdx4;
  integrator->getGradient(2, &dfdx3);
  integrator->getGradient(3, &dfdx4);

  TacsScalar *failmeanderiv, *failvarderiv;
  dfdx3->getArray(&failmeanderiv);

  // Find the derivative of variance
  dfdx4->axpy(-2.0*failmean, dfdx3);
  dfdx4->getArray(&failvarderiv);
  
  printf("CS dE{ fail }/dx = %.17e %.17e %.17e \n", 
         RealPart(failmeanderiv[0]), 
         ImagPart(failmean)/dh,
         RealPart(failmeanderiv[0]) - ImagPart(failmean)/dh);
  printf("CS dV{ fail }/dx = %.17e %.17e %.17e \n", 
         RealPart(failvarderiv[0]), 
         ImagPart(failvar)/dh,
         RealPart(failvarderiv[0]) - ImagPart(failvar)/dh);

  // Get the gradient of dispure
  TACSBVec *dfdx5, *dfdx6;
  integrator->getGradient(4, &dfdx5);
  integrator->getGradient(5, &dfdx6);

  TacsScalar *dispmeanderiv, *dispvarderiv;
  dfdx5->getArray(&dispmeanderiv);

  // Find the derivative of variance
  dfdx6->axpy(-2.0*dispmean, dfdx5);
  dfdx6->getArray(&dispvarderiv);
  
  printf("CS dE{ disp }/dx = %.17e %.17e %.17e \n", 
         RealPart(dispmeanderiv[0]), 
         ImagPart(dispmean)/dh,
         RealPart(dispmeanderiv[0]) - ImagPart(dispmean)/dh);
  printf("CS dV{ disp }/dx = %.17e %.17e %.17e \n", 
         RealPart(dispvarderiv[0]), 
         ImagPart(dispvar)/dh,
         RealPart(dispvarderiv[0]) - ImagPart(dispvar)/dh);

  integrator->decref();
  assembler->decref();

  MPI_Finalize();
  return 0;
}
#endif
