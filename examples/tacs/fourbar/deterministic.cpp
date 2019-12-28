#include "TACSAssembler.h"
#include "TACSIntegrator.h"
#include "TACSRigidBody.h"
#include "TACSKinematicConstraints.h"
#include "MITC3.h"
#include "TACSKSFailure.h"
#include "TACSStructuralMass.h"
#include "TACSConstitutiveVerification.h"
#include "TACSElementVerification.h"
#include "SquareSection.h"
#include "deterministic.h"

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
TACSAssembler *four_bar_mechanism( int nA, int nB, int nC ){
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

  // Set the material properties
  TacsScalar density = 7800.0;
  TacsScalar E = 207e9;
  TacsScalar nu = 0.3;
  TacsScalar G = 0.5*E/(1.0 + nu);

  TacsScalar wA = 0.016; //  + 1.0e-30j;
  TacsScalar wB = 0.008; // 
  int wANum = 0, wBNum = 1;

  TACSTimoshenkoConstitutive *stiffA =
    new SquareSection(density, E, G, wA, wANum, axis_A);

  TACSTimoshenkoConstitutive *stiffB =
    new SquareSection(density, E, G, wA, wANum, axis_B);

  TACSTimoshenkoConstitutive *stiffC =
    new SquareSection(density, E, G, wB, wBNum, axis_C);

  // Set up the connectivity
  MITC3 *beamA = new MITC3(stiffA, gravity);
  MITC3 *beamB = new MITC3(stiffB, gravity);
  MITC3 *beamC = new MITC3(stiffC, gravity);

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
    elems[elem] = beamA;
    ptr[elem+1] = ptr[elem] + 3;
    elem++;
  }

  for ( int i = 0; i < nB; i++ ){
    conn[ptr[elem]] = nodesB[2*i];
    conn[ptr[elem]+1] = nodesB[2*i+1];
    conn[ptr[elem]+2] = nodesB[2*i+2];
    elems[elem] = beamB;
    ptr[elem+1] = ptr[elem] + 3;
    elem++;
  }

  for ( int i = 0; i < nC; i++ ){
    conn[ptr[elem]] = nodesC[2*i];
    conn[ptr[elem]+1] = nodesC[2*i+1];
    conn[ptr[elem]+2] = nodesC[2*i+2];
    elems[elem] = beamC;
    ptr[elem+1] = ptr[elem] + 3;
    elem++;
  }

  // Add the connectivities for the constraints
  conn[ptr[elem]] = nodesA[0];
  conn[ptr[elem]+1] = nnodes-4;
  elems[elem] = revDriverA;
  ptr[elem+1] = ptr[elem] + 2;
  elem++;

  conn[ptr[elem]] = nodesA[2*nA];
  conn[ptr[elem]+1] = nodesB[0];
  conn[ptr[elem]+2] = nnodes-3;
  elems[elem] = revB;
  ptr[elem+1] = ptr[elem] + 3;
  elem++;

  conn[ptr[elem]] = nodesC[0];
  conn[ptr[elem]+1] = nodesB[2*nB];
  conn[ptr[elem]+2] = nnodes-2;
  elems[elem] = revC;
  ptr[elem+1] = ptr[elem] + 3;
  elem++;

  conn[ptr[elem]] = nodesC[2*nC];
  conn[ptr[elem]+1] = nnodes-1;
  elems[elem] = revD;
  ptr[elem+1] = ptr[elem] + 2;
  elem++;

  delete [] nodesA;
  delete [] nodesB;
  delete [] nodesC;

  // Create the TACSAssembler object
  TACSAssembler *assembler = new TACSAssembler(MPI_COMM_WORLD, 8, nnodes, nelems);

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

int main( int argc, char *argv[] ){
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Create the finite-element model
  int nA = 4, nB = 8, nC = 4;
  TACSAssembler *assembler = four_bar_mechanism(nA, nB, nC);
  assembler->incref();

  // Set the final time
  double tf = 12.0;

  // The number of total steps (100 per second)
  int num_steps = 1200;

  // Create the integrator class
  TACSIntegrator *integrator = new TACSBDFIntegrator(assembler, 0.0, tf, num_steps, 2);
  integrator->incref();

  // Set the integrator options
  integrator->setUseSchurMat(1, TACSAssembler::TACS_AMD_ORDER);
  integrator->setAbsTol(1e-7);
  integrator->setRelTol(1e-12);
  integrator->setOutputFrequency(0);
  integrator->setPrintLevel(2);

  // Integrate the equations of motion forward in time
  integrator->integrate();

  // Create the continuous KS function
  double ksRho = 10000.0;
  TACSKSFailure *ksfunc = new TACSKSFailure(assembler, ksRho);
  TACSStructuralMass *fmass = new TACSStructuralMass(assembler);

  // Set the functions
  const int num_funcs = 2;
  TACSFunction **funcs = new TACSFunction*[num_funcs];
  funcs[0] = fmass;
  funcs[1] = ksfunc;
  
  integrator->setFunctions(num_funcs, funcs);

  TacsScalar fval[2];
  integrator->evalFunctions(fval);
  printf("Function value  : %15.10e\n", TacsRealPart(fval[0]));
  printf("Function value  : %15.10e\n", TacsRealPart(fval[1]));

  printf("CSD    value : %15.10e\n", TacsImagPart(fval[0])/1.0e-30);
  printf("CSD    value : %15.10e\n", TacsImagPart(fval[1])/1.0e-30);

  // Evaluate the adjoint
  integrator->integrateAdjoint();

  // Get the gradient
  TACSBVec *dfdx1;
  integrator->getGradient(0, &dfdx1);
  TacsScalar *objgrad;
  dfdx1->getArray(&objgrad);
  printf("adjoint dfdx : %15.10e %15.10e\n", TacsRealPart(objgrad[0]), TacsRealPart(objgrad[1]));
  
  TACSBVec *dfdx2;
  integrator->getGradient(1, &dfdx2);
  TacsScalar *congrad;
  dfdx2->getArray(&congrad);
  printf("adjoint dfdx : %15.10e %15.10e\n", TacsRealPart(congrad[0]), TacsRealPart(congrad[1]));

#ifdef TACS_USE_COMPLEX
  integrator->checkGradients(1e-30);
#else
  integrator->checkGradients(1e-6);
#endif // TACS_USE_COMPLEX

  // Set the output options/locations
  int elem[3];
  elem[0] = nA/2;
  elem[1] = nA + nB/2;
  elem[2] = nA + nB + nC/2;
  double param[][1] = {{-1.0}, {-1.0}, {0.0}};

  // Extra the data to a file
  for ( int pt = 0; pt < 3; pt++ ){
    char filename[128];
    sprintf(filename, "mid_beam_%d.dat", pt+1);
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "Variables = t, u0, v0, w0, quantity\n");

    // Write out data from the beams
    TACSBVec *q = NULL;
    for ( int k = 0; k < num_steps+1; k++ ){
      TacsScalar X[3*3], vars[8*3], dvars[8*3], ddvars[8*3];
      double time = integrator->getStates(k, &q, NULL, NULL);
      assembler->setVariables(q);
      TACSElement *element = assembler->getElement(elem[pt], X, vars, dvars, ddvars);

      TacsScalar quantity;
      element->evalPointQuantity(elem[pt], TACS_FAILURE_INDEX, time,
                                 0, param[pt], X, vars, dvars, ddvars, &quantity);

      fprintf(fp, "%e  %e %e %e  %e\n",
              time, TacsRealPart(vars[0]), TacsRealPart(vars[1]),
              TacsRealPart(vars[2]), TacsRealPart(quantity));
    }
    fclose(fp);
  }

  integrator->decref();
  assembler->decref();

  MPI_Finalize();
  return 0;
}
