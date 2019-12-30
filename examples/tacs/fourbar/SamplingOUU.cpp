#include <cassert>
#include "SamplingOUU.hpp"
#include "sampling.h"
#include "TACSKSFailure.h"
#include "TACSStructuralMass.h"
#include "ParameterFactory.h"

SamplingOUU::SamplingOUU( int _nA, int _nB, int _nC, double _tf, int _num_steps,
                          double _abstol, double _reltol, ParameterContainer *_pc, 
                          double _alpha, double _beta ){
  nA        = _nA;
  nB        = _nB;
  nC        = _nC;
  tf        = _tf;
  num_steps = _num_steps;
  abstol    = _abstol;
  reltol    = _reltol;
  pc        = _pc;
  alpha     = _alpha;
  beta      = _beta;
}

SamplingOUU::~SamplingOUU(){
  delete [] fvals;
  delete [] dfdx;
  delete [] dgdx;
}

void SamplingOUU::evaluateFuncGrad( Index n, const Number* x ){

  int nqpoints = 5;
  int nqpts[1] = {nqpoints};

  // Store mass, failure, mass deriv, failure deriv
  TacsScalar **data = new TacsScalar*[nqpoints];
  for (int i = 0; i < nqpoints; i++){
    data[i] = new TacsScalar[6];
  }

  const int nvars = pc->getNumParameters();
  TacsScalar *zq = new TacsScalar[nvars];
  TacsScalar *yq = new TacsScalar[nvars];
  TacsScalar wq;

  for (int iq = 0; iq < nqpoints; iq++){

    wq = pc->quadrature(iq, zq, yq);

    // Create the finite-element model
    TACSAssembler *assembler = four_bar_mechanism(nA, nB, nC, yq[0]);
    assembler->incref();

    // Create the integrator class
    TACSIntegrator *integrator =
      new TACSBDFIntegrator(assembler, 0.0, tf, num_steps, 2);
    integrator->incref();

    // Set the integrator options
    integrator->setUseSchurMat(1, TACSAssembler::TACS_AMD_ORDER);
    integrator->setAbsTol(abstol);
    integrator->setRelTol(reltol);
    integrator->setOutputFrequency(0);

    // Integrate the equations of motion forward in time
    integrator->integrate();

    // Create the continuous KS function
    double ksRho = 10000.0;
    TACSKSFailure *ksfunc = new TACSKSFailure(assembler, ksRho);
    TACSStructuralMass *fmass = new TACSStructuralMass(assembler);

    // Set the functions
    const int num_funcs = 2;
    TACSFunction **funcs = new TACSFunction*[num_funcs]; //fmass
    funcs[0] = fmass;
    funcs[1] = ksfunc;
    integrator->setFunctions(num_funcs, funcs);

    TacsScalar ftmp[num_funcs];
    integrator->evalFunctions(ftmp);

    // Evaluate the adjoint
    integrator->integrateAdjoint();

    // Get the gradient
    TACSBVec *massdfdx;
    TACSBVec *faildfdx;
    integrator->getGradient(0, &massdfdx);
    integrator->getGradient(1, &faildfdx);

    TacsScalar *massdfdxvals, *faildfdxvals;
    massdfdx->getArray(&massdfdxvals);
    faildfdx->getArray(&faildfdxvals);

    data[iq][0] = ftmp[0];
    data[iq][1] = ftmp[1];

    data[iq][2] = massdfdxvals[0];
    data[iq][3] = massdfdxvals[1];

    data[iq][4] = faildfdxvals[0];
    data[iq][5] = faildfdxvals[1];
    
    integrator->decref();
    assembler->decref();

  } // end quadrature

  TacsScalar failmean, fail2mean, failvar, failstd;
  TacsScalar massmean, mass2mean, massvar, massstd;

  TacsScalar massmeanderiv[2], massderivtmp[2], massvarderiv[2], massstdderiv[2];
  TacsScalar failmeanderiv[2], failderivtmp[2], failvarderiv[2], failstdderiv[2];

  // Compute mean and variance of mid point of beam 1
  for (int q = 0; q < nqpoints; q++){

    wq = pc->quadrature(q, zq, yq);

    // mass 
    massmean += wq*data[q][0]; // E[F]
    mass2mean += wq*data[q][0]*data[q][0]; // E{F*F}

    // failure
    failmean += wq*data[q][1]; // E[F]
    fail2mean += wq*data[q][1]*data[q][1]; // E{F*F}

    // massmean deriv
    massmeanderiv[0] += wq*data[q][2]; // E[F]
    massderivtmp[0]  += wq*2.0*data[q][0]*data[q][2]; // E{F*F}

    massmeanderiv[1] += wq*data[q][3]; // E[F]
    massderivtmp[1]  += wq*2.0*data[q][0]*data[q][3]; // E{F*F}

    failmeanderiv[0] += wq*data[q][4]; // E[F]
    failderivtmp[0]  += wq*2.0*data[q][1]*data[q][4]; // E{F*F}

    failmeanderiv[1] += wq*data[q][5]; // E[F]
    failderivtmp[1]  += wq*2.0*data[q][1]*data[q][5]; // E{F*F}
 
  }
  
  // Compute mean and variance
  massvar = mass2mean - massmean*massmean;
  failvar = fail2mean - failmean*failmean;

  massvarderiv[0] = massderivtmp[0] - 2.0*massmean*massmeanderiv[0];
  massvarderiv[1] = massderivtmp[1] - 2.0*massmean*massmeanderiv[1];

  failvarderiv[0] = failderivtmp[0] - 2.0*failmean*failmeanderiv[0];
  failvarderiv[1] = failderivtmp[1] - 2.0*failmean*failmeanderiv[1];

  if (abs(massvar) > 1.0e-8){
    massstdderiv[0] = massvarderiv[0]/(2.0*massstd);
    massstdderiv[1] = massvarderiv[1]/(2.0*massstd);
  } else {
    printf("small variance of mass %e \n", massvar);
    massstdderiv[0] = 0.0;
    massstdderiv[1] = 0.0;
  }

  if (abs(failvar) > 1.0e-8){
    failstdderiv[0] = failvarderiv[0]/(2.0*failstd);
    failstdderiv[1] = failvarderiv[1]/(2.0*failstd);
  } else {
    printf("small variance of fail %e \n", failvar);
    failstdderiv[0] = 0.0;
    failstdderiv[1] = 0.0;
  }

  // Store the evaluated function values
  this->fvals[0] = massmean + massstd;
  this->fvals[1] = failmean + this->beta*failstd;

  // Objective function gradient
  for (int i = 0; i < n; i++){
    this->dfdx[i] = massmeanderiv[i] + massstdderiv[i];
  }

  // constraint function gradient
  for (int i = 0; i < n; i++){
    this->dgdx[i] = failmeanderiv[i] + this->beta*failstdderiv[i];
  }

  for (int i = 0; i < nqpoints; i++){
    delete [] data[i];
  }

}

bool SamplingOUU::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                               Index& nnz_h_lag, IndexStyleEnum& index_style)
{
  // The problem described in SamplingOUU.hpp has 2 variables, x1, & x2,
  n = 2;

  // one equality constraint,
  m = 1;

  // 2 nonzeros in the jacobian (one for x1, and one for x2),
  nnz_jac_g = m*n;

  // and 2 nonzeros in the hessian of the lagrangian
  // (one in the hessian of the objective for x2,
  //  and one in the hessian of the constraints for x1)
  nnz_h_lag = n*(n+1)/2;

  // We use the standard fortran index style for row/col entries
  index_style = FORTRAN_STYLE;

  // space to store function and gradient values
  y     = new double[n];    
  fvals = new double[m+1];
  dfdx  = new double[n];
  dgdx  = new double[n];
  
  return true;
}

bool SamplingOUU::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                  Index m, Number* g_l, Number* g_u)
{
  assert(n == 2);
  assert(m == 1);

  // set upper and lower bounds on DV
  for (int i = 0; i < n; i++){
    x_l[i] = 0.005;
    x_u[i] = 0.020;
  }

  // Set bounds on inequatlity constraint
  g_l[0] = -2.0e9;
  g_u[0] = 1.0;

  return true;
}

bool SamplingOUU::get_starting_point(Index n, bool init_x, Number* x,
                                     bool init_z, Number* z_L, Number* z_U,
                                     Index m, bool init_lambda,
                                     Number* lambda)
{
  // Here, we assume we only have starting values for x, if you code
  // your own NLP, you can provide starting values for the others if
  // you wish.
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);

  // we initialize x in bounds, in the upper right quadrant
  for (int i = 0; i < n; i++){
    x[i] = 0.010;
  }

  return true;
}

bool SamplingOUU::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{

  if (new_x){
    this->evaluateFuncGrad(n, x);
  }
  
  obj_value = this->fvals[0];
  
  return true;
}

bool SamplingOUU::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{

  if (new_x){
    this->evaluateFuncGrad(n, x);
  }

  for (int i = 0; i < n; i++){  
    grad_f[i] = this->dfdx[i];
  }

  return true;
}

bool SamplingOUU::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
  
  if (new_x){
    this->evaluateFuncGrad(n, x);
  }

  g[0] = this->fvals[1];

  return true;
}

bool SamplingOUU::eval_jac_g(Index n, const Number* x, bool new_x,
                             Index m, Index nele_jac, Index* iRow, Index *jCol,
                             Number* values)
{
  if (values == NULL) {
    // return the structure of the jacobian of the constraints

    // element at 1,1: grad_{x1} g_{1}(x)
    iRow[0] = 1;
    jCol[0] = 1;

    // element at 1,2: grad_{x2} g_{1}(x)
    iRow[1] = 1;
    jCol[1] = 2;
  }
  else {

    if (new_x){
      this->evaluateFuncGrad(n, x);
    }
    
    for (int i = 0; i < n; i++){
      values[i] = this->dgdx[i];
    }

  }

  return true;
}

bool SamplingOUU::eval_h(Index n, const Number* x, bool new_x,
                         Number obj_factor, Index m, const Number* lambda,
                         bool new_lambda, Index nele_hess, Index* iRow,
                         Index* jCol, Number* values)
{

  if (values == NULL) {

    // return the structure. This is a symmetric matrix, fill the lower left
    // triangle only.

    for (int i = 0; i < n; i++){
      iRow[i] = i + 1;
      jCol[i] = i + 1;
    }

    // Note: off-diagonal elements are zero for this problem
  }
  else {

    // return the values
    std::cout<< "WARNING:Hessian CALL"<<std::endl;
    exit(-1);
     
  }

  return true;
}

void SamplingOUU::finalize_solution(SolverReturn status,
                                    Index n, const Number* x, const Number* z_L, const Number* z_U,
                                    Index m, const Number* g, const Number* lambda,
                                    Number obj_value,
                                    const IpoptData* ip_data,
                                    IpoptCalculatedQuantities* ip_cq)
{

  // here is where we would store the solution to variables, or write to a file, etc
  // so we could use the solution.
  // For this example, we write the solution to the console
  std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
  for( Index i = 0; i < n; i++ )
    {
      std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }
  std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
  for( Index i = 0; i < n; i++ )
    {
      std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
    }
  for( Index i = 0; i < n; i++ )
    {
      std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
    }
  std::cout << std::endl << std::endl << "Objective value" << std::endl;
  std::cout << "f(x*) = " << obj_value << std::endl;
  std::cout << std::endl << "Final value of the constraints:" << std::endl;
  for( Index i = 0; i < m; i++ )
    {
      std::cout << "g(" << i << ") = " << g[i] << std::endl;
    }
}

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"

int main(int argc, char *argv[] ){

  MPI_Init(&argc, &argv);

  ParameterFactory *factory  = new ParameterFactory();
  AbstractParameter *ptheta = factory->createNormalParameter(5.0, 2.5, 4);
  ParameterContainer *pc = new ParameterContainer();
  pc->addParameter(ptheta);  
  pc->initialize();

  const int nsterms  = pc->getNumBasisTerms();
  const int nsqpts   = pc->getNumQuadraturePoints();

  double alpha = 0.5; // objective robustness
  double beta  = 3.0;  // constraint reliability
  int nA = 4, nB = 8, nC = 4;
  double tf = 12.0;
  int num_steps = 12000;
  double abstol = 1.0e-7;
  double reltol = 1.0e-12;
  SmartPtr<TNLP> mynlp = new SamplingOUU(nA, nB, nC, tf, num_steps, 
                                         abstol, reltol, pc,
                                         alpha, beta);
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

  // Change some options
  app->Options()->SetStringValue("mu_strategy", "adaptive");
  app->Options()->SetStringValue("output_file", "deterministic_ipopt.out");
  app->Options()->SetIntegerValue("max_iter", 100);
  app->Options()->SetStringValue("hessian_approximation", "limited-memory");
  app->Options()->SetStringValue("limited_memory_update_type", "bfgs");

  app->Options()->SetNumericValue("tol", 1e-6);
  app->Options()->SetNumericValue("constr_viol_tol", 0.0001);
  app->Options()->SetNumericValue("acceptable_tol", 1.0e-3);
  
  app->Options()->SetStringValue("derivative_test", "first-order");
  app->Options()->SetNumericValue("derivative_test_perturbation", 1.0e-6);
  app->Options()->SetStringValue("derivative_test_print_all", "yes");
    
  // The following overwrites the default name (ipopt.opt) of the options file
  // app->Options()->SetStringValue("option_file_name", "hs071.opt");
  // Initialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if( status != Solve_Succeeded )
    {
      std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
      return (int) status;
    }
  // Ask Ipopt to solve the problem
  status = app->OptimizeTNLP(mynlp);
  if( status == Solve_Succeeded )
    {
      std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
    }
  else
    {
      std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
    }
  // As the SmartPtrs go out of scope, the reference count
  // will be decremented and the objects will automatically
  // be deleted.
   
  MPI_Finalize();

  return (int) status;
}
