#include "TimoshenkoStiffness.h"
#include "MITC3.h"
#include "TACSAssembler.h"
#include "RigidBody.h"
#include "TACSIntegrator.h"
#include "KinematicConstraints.h"
#include "ParameterContainer.h"
#include "ParameterFactory.h"
#include "TACSStochasticElement.h"


void updateRevDriver( TACSElement *elem, TacsScalar *vals ){
  TACSRevoluteDriver *driver = dynamic_cast<TACSRevoluteDriver*>(elem);
  if (driver != NULL) {
    driver->setSpeed(vals[2]);
  } else {
    printf("Element mismatch while updating...");
  }
}

void updateBeamAB( TACSElement *elem, TacsScalar *vals ){
  MITC3 *mitc3 = dynamic_cast<MITC3*>(elem);
  if (mitc3 != NULL) {
    // Get the constitute object
    TimoshenkoStiffness *stiff = dynamic_cast<TimoshenkoStiffness*>(mitc3->getConstitutive());
    if (stiff){
      stiff->incref();
      stiff->rho[0] = vals[0];
      stiff->decref();        
    }
    // Check if values are set properly
    TimoshenkoStiffness *stiff2 =  dynamic_cast<TimoshenkoStiffness*>(mitc3->getConstitutive());    
    if (stiff2){
      stiff2->incref();      
      //printf("set value is %e %e\n", vals[0], stiff2->rho[0]);       
      stiff2->decref();
    }
  } else {
    printf("Element mismatch while updating...");
  }
}

void updateBeamC( TACSElement *elem, TacsScalar *vals ){
  MITC3 *mitc3 = dynamic_cast<MITC3*>(elem);
  if (mitc3 != NULL) {
    // Get the constitute object
    TimoshenkoStiffness *stiff = dynamic_cast<TimoshenkoStiffness*>(mitc3->getConstitutive());
    if (stiff){
      stiff->incref();
      stiff->rho[0] = vals[1];
      stiff->decref();        
    }
    // Check if values are set properly
    TimoshenkoStiffness *stiff2 =  dynamic_cast<TimoshenkoStiffness*>(mitc3->getConstitutive());    
    if (stiff2){
      stiff2->incref();      
      //printf("set value is %e %e\n", vals[0], stiff2->rho[0]);       
      stiff2->decref();
    }
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
  int fixed_point = 1;  int not_fixed = 0;  
  TACSRevoluteDriver *revDriverA = new TACSRevoluteDriver(revDirA, omega);
  TACSRevoluteConstraint *revB   = new TACSRevoluteConstraint(not_fixed, ptB, revDirB);
  TACSRevoluteConstraint *revC   = new TACSRevoluteConstraint(not_fixed, ptC, revDirC);
  TACSRevoluteConstraint *revD   = new TACSRevoluteConstraint(fixed_point, ptD, revDirD);

  // Create the stiffness objects for each element
  TacsScalar mA = 1.997; // kg/m
  TacsScalar IA = 42.60e-6; // kg*m

  TacsScalar EA_A = 52.99e6;
  TacsScalar GJ_A = 733.5;
  TacsScalar kGAz_A = 16.88e6; 
  TacsScalar EIz_A = 1131.0;

  // The properties of the second beam
  TacsScalar mB = 0.4992; // kg*m^2/m
  TacsScalar IB = 2.662e-6; // kg*m^2/m

  TacsScalar EA_B = 13.25e6;
  TacsScalar GJ_B = 45.84;
  TacsScalar kGAz_B = 4.220e6;
  TacsScalar EIz_B = 70.66;

  // Set the reference axes for each beam
  TacsScalar axis_A[] = {-1.0, 0.0, 0.0};
  TacsScalar axis_B[] = {0.0, 1.0, 0.0};
  TacsScalar axis_C[] = {1.0, 0.0, 0.0};
  
  // Create the Timoshenko stiffness object
  TimoshenkoStiffness *stiffA =
    new TimoshenkoStiffness(mA, IA, IA, 0.0,
                            EA_A, GJ_A, EIz_A, EIz_A, kGAz_A, kGAz_A,
                            axis_A);

  TimoshenkoStiffness *stiffB =
    new TimoshenkoStiffness(mA, IA, IA, 0.0,
                            EA_A, GJ_A, EIz_A, EIz_A, kGAz_A, kGAz_A,
                            axis_B);

  TimoshenkoStiffness *stiffC =
    new TimoshenkoStiffness(mB, IB, IB, 0.0,
                            EA_B, GJ_B, EIz_B, EIz_B, kGAz_B, kGAz_B,
                            axis_C);

  // Set up the connectivity
  MITC3 *beamA = new MITC3(stiffA, gravity);
  MITC3 *beamB = new MITC3(stiffB, gravity);
  MITC3 *beamC = new MITC3(stiffC, gravity);

  ParameterFactory *factory = new ParameterFactory();
  AbstractParameter *pmA = factory->createExponentialParameter(mA, 0.1, 2);
  AbstractParameter *pmC = factory->createExponentialParameter(mB, 0.1, 0);
  AbstractParameter *pspeed = factory->createUniformParameter(-0.7, -0.5, 0);

  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(pmA);
  pc->addParameter(pmC);
  pc->addParameter(pspeed);
  pc->initialize();
  
  int nsterms = pc->getNumBasisTerms();
  printf("nsterms = %d \n", nsterms);

  // Stochastic elements
  TACSStochasticElement *sbeamA = new TACSStochasticElement(beamA, pc);
  sbeamA->incref();
  sbeamA->setUpdateCallback(updateBeamAB);

  TACSStochasticElement *sbeamB = new TACSStochasticElement(beamB, pc);
  sbeamB->incref();
  sbeamB->setUpdateCallback(updateBeamAB);

  TACSStochasticElement *sbeamC = new TACSStochasticElement(beamC, pc);
  sbeamC->incref();
  sbeamC->setUpdateCallback(updateBeamC);

  TACSStochasticElement *srevDriverA = new TACSStochasticElement(revDriverA, pc);
  srevDriverA->setUpdateCallback(updateRevDriver);
  
  TACSStochasticElement *srevB = new TACSStochasticElement(revB, pc);
  TACSStochasticElement *srevC = new TACSStochasticElement(revC, pc);
  TACSStochasticElement *srevD = new TACSStochasticElement(revD, pc);

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

  // Create the TACSAssembler object
  int vars_per_node = 8*nsterms;
  TACSAssembler *tacs = new TACSAssembler(MPI_COMM_WORLD, vars_per_node, nnodes, nelems);

  tacs->setElementConnectivity(conn, ptr);
  delete [] conn;
  delete [] ptr;

  tacs->setElements(elems);
  delete [] elems;

  tacs->initialize();

  // Set the node locations
  TACSBVec *Xvec = tacs->createNodeVec();
  Xvec->incref();
  TacsScalar *Xarray;
  Xvec->getArray(&Xarray);
  memcpy(Xarray, X, 3*nnodes*sizeof(TacsScalar));
  tacs->setNodes(Xvec);
  Xvec->decref();
  delete [] X;

  return tacs;
}

int main( int argc, char *argv[] ){
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Create the finite-element model
  int nA = 1, nB = 1, nC = 1;
  TACSAssembler *tacs = four_bar_mechanism(nA, nB, nC);
  tacs->incref();

  // Set the final time
  double tf = 1.0e-3;

  // The number of total steps (100 per second)
  int num_steps = 1;

  // Create the integrator class
  TACSIntegrator *integrator = new TACSBDFIntegrator(tacs, 0.0, tf, num_steps, 2);
  integrator->incref();

  // Set the integrator options
  integrator->setUseFEMat(1, TACSAssembler::NATURAL_ORDER);
  integrator->setAbsTol(1e-7);
  integrator->setOutputFrequency(10);
  integrator->setPrintLevel(2);
  integrator->integrate();

  return 0;
  integrator->decref();
  tacs->decref();

  MPI_Finalize();
  return 0;
  
  // Set the output options/locations
  int elem[3];
  // elem[0] = nA/2;
  // elem[1] = nA + nB/2;
  // elem[2] = nA + nB + nC/2;
  // double param[][1] = {{-1.0}, {-1.0}, {-1.0}}; 
  elem[0] = nA/2;
  elem[1] = nA + nB/2;
  elem[2] = nA + nB + nC/2;
  double param[][1] = {{-1.0}, {-1.0}, {0.0}}; 

  // Extra the data to a file
  for ( int pt = 0; pt < 3; pt++ ){
    char filename[128];
    sprintf(filename, "mid_beam_%d.dat", pt+1);
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "Variables = t, u0, v0, w0, sx0, st0, sy1, sz1, sxy0, sxz0\n");

    // Write out data from the beams
    TACSBVec *q = NULL;
    for ( int k = 0; k < num_steps+1; k++ ){
      TacsScalar X[3*3], vars[8*3];
      double time = integrator->getStates(k, &q, NULL, NULL);
      tacs->setVariables(q);
      TACSElement *element = tacs->getElement(elem[pt], X, vars);

      TacsScalar e[6], s[6];
      element->getStrain(e, param[pt], X, vars);
      TACSConstitutive *con = element->getConstitutive();
      con->calculateStress(param[pt], e, s);

      fprintf(fp, "%e  %e %e %e  %e %e %e  %e %e %e\n",
              time, vars[0], vars[1], vars[2], 
              s[0], s[1], s[2], s[3], s[4], s[5]);
    }
    fclose(fp);
  }

}
