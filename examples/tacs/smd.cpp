#include "smd.h"

#include "TACSCreator.h"
#include "TACSAssembler.h"
#include "TACSIntegrator.h"
#include "TACSFunction.h"

#include "ParameterContainer.h"
#include "ParameterFactory.h"

#include "TACSStochasticElement.h"
#include "TACSStochasticFunction.h"
#include "TACSKSStochasticFunction.h"
#include "TACSKSFunction.h"

#include "TACSKineticEnergy.h"
#include "TACSPotentialEnergy.h"
#include "TACSDisplacement.h"
#include "TACSVelocity.h"

void updateSMD( TACSElement *elem, TacsScalar *vals ){
  SMD *smd = dynamic_cast<SMD*>(elem);
  if (smd != NULL) {
    //smd->m = vals[0];
    smd->c = vals[0];
    // smd->k = vals[0];
    // printf("%e %e %e \n", smd->m, smd->c, smd->k);
  } else {
    printf("Element mismatch while updating...");
  }
}

SMD::SMD(double m, double c, double k){
  this->m = m;
  this->c = c;
  this->k = k;  
}

void SMD::getInitConditions( int elemIndex, const TacsScalar X[],
                             TacsScalar v[], TacsScalar dv[], TacsScalar ddv[] ){
  int num_vars = getNumNodes()*getVarsPerNode();
  memset(v, 0, num_vars*sizeof(TacsScalar));
  memset(dv, 0, num_vars*sizeof(TacsScalar));
  memset(ddv, 0, num_vars*sizeof(TacsScalar));

  // set init conditions
  v[0] = 1.0;
  dv[0] = 0.0;
}

void SMD::addResidual( int elemIndex, double time,
                       const TacsScalar X[], const TacsScalar v[],
                       const TacsScalar dv[], const TacsScalar ddv[],
                       TacsScalar res[] ){
  res[0] += m*ddv[0] + c*dv[0] + k*v[0];
}

void SMD::addJacobian( int elemIndex, double time,
                       TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                       const TacsScalar X[], const TacsScalar v[],
                       const TacsScalar dv[], const TacsScalar ddv[],
                       TacsScalar res[], TacsScalar mat[] ){
  addResidual(elemIndex, time, X, v, dv, ddv, res);
  mat[0] += gamma*m + beta*c + alpha*k;
}

int SMD::evalPointQuantity( int elemIndex, int quantityType,
                            double time,
                            int n, double pt[],
                            const TacsScalar Xpts[],
                            const TacsScalar v[],
                            const TacsScalar dv[],
                            const TacsScalar ddv[],
                            TacsScalar *quantity ){
  if (quantityType == TACS_KINETIC_ENERGY_FUNCTION){
    *quantity = 0.5*m*dv[0]*dv[0];
    return 1;
  } else  if (quantityType == TACS_POTENTIAL_ENERGY_FUNCTION){
    *quantity = 0.5*k*v[0]*v[0];
    return 1;
  } else  if (quantityType == TACS_DISPLACEMENT_FUNCTION){
    *quantity = v[0];
    return 1;
  } else  if (quantityType == TACS_VELOCITY_FUNCTION){
    *quantity = dv[0];
    return 1;
  }
  return 0;
}

void SMD::addPointQuantitySVSens( int elemIndex, int quantityType,
                                  double time,
                                  TacsScalar alpha,
                                  TacsScalar beta,
                                  TacsScalar gamma,
                                  int n, double pt[],
                                  const TacsScalar Xpts[],
                                  const TacsScalar v[],
                                  const TacsScalar dv[],
                                  const TacsScalar ddv[],
                                  const TacsScalar dfdq[],
                                  TacsScalar dfdu[] ){
  if (quantityType == TACS_KINETIC_ENERGY_FUNCTION){
    dfdu[0] += beta*m*dv[0];
  } else  if (quantityType == TACS_POTENTIAL_ENERGY_FUNCTION){
    dfdu[0] += alpha*k*v[0];
  } else  if (quantityType == TACS_DISPLACEMENT_FUNCTION){
    dfdu[0] += alpha;
  } else  if (quantityType == TACS_VELOCITY_FUNCTION){
    dfdu[0] += beta;
  }
}

/*
  Implementation for dFdx
*/
void SMD::addPointQuantityDVSens( int elemIndex, int quantityType,
                                  double time,
                                  TacsScalar scale,
                                  int n, double pt[],
                                  const TacsScalar Xpts[],
                                  const TacsScalar v[],
                                  const TacsScalar dv[],
                                  const TacsScalar ddv[],
                                  const TacsScalar dfdq[],
                                  int dvLen,
                                  TacsScalar dfdx[] ){
  //  printf("addPointQuantityDVSens \n");
  // assuming 'm' as the design variable 0 and 'k' as design variable 1
  if (quantityType == TACS_KINETIC_ENERGY_FUNCTION){
    dfdx[0] += scale*0.5*dv[0]*dv[0];
    dfdx[1] += 0.0;
  } else  if (quantityType == TACS_POTENTIAL_ENERGY_FUNCTION){
    dfdx[0] += 0.0;
    dfdx[1] += scale*0.5*v[0]*v[0];
  } else  if (quantityType == TACS_DISPLACEMENT_FUNCTION){
    dfdx[0] += 0.0;
    dfdx[1] += 0.0;
  } else  if (quantityType == TACS_VELOCITY_FUNCTION){
    dfdx[0] += 0.0;
    dfdx[1] += 0.0;
  }
}

/*
  Adjoint residual product
*/
void SMD::addAdjResProduct( int elemIndex, double time,
                            TacsScalar scale,
                            const TacsScalar psi[],
                            const TacsScalar Xpts[],
                            const TacsScalar v[],
                            const TacsScalar dv[],
                            const TacsScalar ddv[],
                            int dvLen, 
                            TacsScalar dfdx[] ){
  // printf("enters addAdjResProduct \n");
  // printf("adjoint is = %e \n", psi[0]);
  // printf("dvlen is = %d \n", dvLen);

  dfdx[0] += scale*psi[0]*ddv[0];
  dfdx[1] += scale*psi[0]*v[0];
 
}

int main( int argc, char *argv[] ){  

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank; 
  MPI_Comm_rank(comm, &rank); 

  //-----------------------------------------------------------------//
  // Choose solution mode (sampling = 0 or 1)
  //-----------------------------------------------------------------//
  
  int sampling = 1;
  int ks = 0; 

  //-------------------------------------------------1----------------//
  // Define random parameters with distribution functions
  //-----------------------------------------------------------------//
  ParameterFactory *factory = new ParameterFactory();
  //AbstractParameter *m = factory->createExponentialParameter(2.5, 0.5, 8);
  AbstractParameter *c = factory->createNormalParameter(0.2, 0.01, 4);
  //AbstractParameter *k = factory->createNormalParameter(5.0, 0.1, 4);
 
  ParameterContainer *pc = new ParameterContainer();
  //pc->addParameter(m);
  pc->addParameter(c);
  //pc->addParameter(k);
  
  if (sampling){
    printf("initializing quadrature\n");
    int nqpts[3] = {1, 9, 9};
    pc->initializeQuadrature(nqpts);
  } else {
    pc->initialize();
  }
  
  int nsterms = pc->getNumBasisTerms();
  if (!sampling){
    printf("nsterms = %d \n", nsterms);
  }
  
  const int num_funcs = 2;
  int nqpoints = pc->getNumQuadraturePoints();
  const int nvars = pc->getNumParameters();
  if (!sampling) nqpoints = 1;

  TacsScalar *fmean = new TacsScalar[ num_funcs ];
  memset(fmean, 0, num_funcs*sizeof(TacsScalar));

  TacsScalar *fvals = new TacsScalar[ num_funcs*nqpoints ];
  memset(fvals, 0, num_funcs*nqpoints*sizeof(TacsScalar));

  TacsScalar *fvar = new TacsScalar[ num_funcs ];
  memset(fvar, 0, num_funcs*sizeof(TacsScalar));

  TacsScalar *ftmp = new TacsScalar[ num_funcs ];
  memset(ftmp, 0, num_funcs*sizeof(TacsScalar));  
  
  double *zq = new double[nvars];
  double *yq = new double[nvars];
  double wq;

  //-----------------------------------------------------------------//
  // Create sampling and stochastic elements
  //-----------------------------------------------------------------//
  
  for (int q = 0; q < nqpoints; q++){

    wq = pc->quadrature(q, zq, yq);
   
    TACSElement *smd = new SMD(2.5, 0.2, 5.0); 
    smd->incref();

    TACSStochasticElement *ssmd = new TACSStochasticElement(smd, pc, updateSMD);
    ssmd->incref();

    int nelems = 1;
    int nnodes = 1;  
    int vars_per_node = 1;
    if (!sampling){
      vars_per_node *= nsterms;
    }

    // Array of elements
    TACSElement **elems = new TACSElement*[nelems];
    if (sampling){
      elems[0] = smd;
    } else{
      elems[0] = ssmd;
    }

    if (sampling){
      updateSMD(smd, yq);
    }
  
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
    creator->setElements(nelems, elems);

    TACSAssembler *assembler = creator->createTACS();
    assembler->incref();
    creator->decref();

    delete [] X;
    delete [] ptr;
    delete [] eids;
    delete [] conn;
    delete [] elems;
  
    // Deterministic Functions
    TACSFunction *ke, *pe, *disp, *vel;    
    TACSFunction *ske, *spe, *sdisp, *svel;    

    double ksweight = 50000.0;    
    if (ks){
      pe    = new TACSKSFunction(assembler, TACS_POTENTIAL_ENERGY_FUNCTION, ksweight);
      disp  = new TACSKSFunction(assembler, TACS_DISPLACEMENT_FUNCTION, ksweight);

      if (!sampling){
        spe   = new TACSKSStochasticFunction(pe, TACS_POTENTIAL_ENERGY_FUNCTION, ksweight, pc, 0);
        sdisp = new TACSKSStochasticFunction(disp, TACS_DISPLACEMENT_FUNCTION, ksweight, pc, 0);
      }

    } else {

      pe    = new TACSPotentialEnergy(assembler); 
      disp  = new TACSDisplacement(assembler); 

      if (!sampling){
        spe   = new TACSStochasticFunction(assembler, pe, pc, TACS_POTENTIAL_ENERGY_FUNCTION);
        sdisp = new TACSStochasticFunction(assembler, disp, pc, TACS_DISPLACEMENT_FUNCTION);
      }

    }

    // Create an array of functions for TACS to evaluate
    TACSFunction **funcs = new TACSFunction*[num_funcs];
    if (sampling){
      // Sample deterministic functions to evaluate moments
      funcs[0] = pe;
      funcs[1] = disp;    
    } else {
      // Decompose stochastic functions to evaluate moments      
      funcs[0] = spe;
      funcs[1] = sdisp;
    }

    //---------------------------------------------------------------//
    // Create the integrator class
    //---------------------------------------------------------------//
    
    TACSIntegrator *bdf = new TACSBDFIntegrator(assembler, 0.0, 10.0, 100, 2);
    bdf->incref();
    bdf->setAbsTol(1e-12);
    bdf->setPrintLevel(0);
    bdf->setFunctions(num_funcs, funcs);
    bdf->integrate();
    bdf->evalFunctions(ftmp);
    bdf->integrateAdjoint();
    
    TACSBVec *dfdx1 = assembler->createDesignVec();
    bdf->getGradient(0, &dfdx1);
    TacsScalar *dfdx1vals;
    dfdx1->getArray(&dfdx1vals);
    printf("d{pe}dm = %e %e \n", dfdx1vals[0], dfdx1vals[1]);

    TACSBVec *dfdx2 = assembler->createDesignVec();
    bdf->getGradient(1, &dfdx2);
    TacsScalar *dfdx2vals;
    dfdx2->getArray(&dfdx2vals);
    printf("d{u}dk  = %e %e \n", dfdx2vals[0], dfdx2vals[1]);

    // store function values for computing moments
    if (sampling){
      int ptr = q*num_funcs;
      for (int i = 0; i < num_funcs; i++){
        fvals[ptr+i] = ftmp[i];
      }
      for (int i = 0; i < num_funcs; i++){
        printf("f = %e\n", ftmp[i]);
        fmean[i] += wq*ftmp[i];
      }
    }
    
    bdf->decref();
    assembler->decref();

  } // end qloop

  if (sampling){
    
    for (int i = 0; i < num_funcs; i++){
      printf("sampling  E[f] = %e\n", fmean[i]);
    }

    // Compute variance
    for (int q = 0; q < nqpoints; q++){
      wq = pc->quadrature(q, zq, yq);
      for (int i = 0; i < num_funcs; i++){
        fvar[i] += wq*(fmean[i]-fvals[q*num_funcs+i])*(fmean[i]-fvals[q*num_funcs+i]);
      }
    }

    printf("\n");
    for (int i = 0; i < num_funcs; i++){
      printf("sampling  V[f] = %e\n", fvar[i]);
    }

  } else {

    for (int i = 0; i < num_funcs; i++){
      printf("projection E[f] = %e\n", ftmp[i]);
    }

  }
  
  MPI_Finalize();
  return 0;
}
