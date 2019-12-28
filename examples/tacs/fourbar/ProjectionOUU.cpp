#include <cassert>
#include "ProjectionOUU.hpp"
#include "projection.h"
#include "TACSKSFailure.h"
#include "TACSStructuralMass.h"
#include "ParameterFactory.h"
#include "TACSKSStochasticFunction.h"
#include "TACSStochasticFunction.h"

ProjectionOUU::ProjectionOUU( int nA, int nB, int nC, double tf, int num_steps,
                              double abstol, double reltol, ParameterContainer *_pc, 
                              double _alpha, double _beta ){
  // Pointer to parameter container
  pc  = _pc;
  alpha = _alpha;
  beta = _beta;

  // Create the finite-element model
  assembler = four_bar_mechanism(nA, nB, nC, pc);
  assembler->incref();
  
  // Create the integrator class
  // integrator = new TACSBDFIntegrator(assembler, 0.0, tf, num_steps, 2);
  integrator = new TACSDIRKIntegrator(assembler, 0.0, tf, num_steps, 3);
  integrator->incref();

  // Set the integrator options
  integrator->setUseSchurMat(1, TACSAssembler::TACS_AMD_ORDER);
  integrator->setAbsTol(reltol);
  integrator->setRelTol(abstol);
  integrator->setOutputFrequency(0);
  integrator->setPrintLevel(0);

  // Set the functions
  const int num_funcs = 4; // mean and variance
  TACSFunction **funcs = new TACSFunction*[num_funcs];

  // Create the deterministic functions
  double ksRho = 10000.0;
  TACSKSFailure  *ksfunc = new TACSKSFailure(assembler, ksRho);
  TACSStructuralMass *fmass = new TACSStructuralMass(assembler);

  // Create the stochastic functions for mean and variance
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
 
  // Create stochastic functions to set into TACS 
  integrator->setFunctions(num_funcs, funcs);
}

ProjectionOUU::~ProjectionOUU(){
  assembler->decref();
  integrator->decref();
  delete [] fvals;
  delete [] dfdx;
  delete [] dgdx;
}

void ProjectionOUU::evaluateFuncGrad( Index n, const Number* x ){

  printf("TACS evaluation at %e %e \n", x[0], x[1]);

  // Store dvs into class variable
  for (int i = 0; i < n; i++){
    this->y[i] = x[i];
  }

  //----------------------------------------------------------------//
  // Update TACS with new design variables
  //----------------------------------------------------------------//

  TACSBVec *X = assembler->createDesignVec();
  X->incref();
  TacsScalar *xvals;
  X->getArray(&xvals);
  for (int i = 0; i < n; i++){
    xvals[i] = x[i];
  }
  assembler->setDesignVars(X);
  X->decref();

  //----------------------------------------------------------------//
  // setup function and constraint values
  //----------------------------------------------------------------//

  // Perform forward TACS Solve
  integrator->integrate();

  TacsScalar ftmp[4];
  integrator->evalFunctions(ftmp);   

  TacsScalar massmean = ftmp[0];
  TacsScalar massvar  = ftmp[1];
  TacsScalar massstd  = sqrt(massvar);
  TacsScalar failmean = ftmp[2];
  TacsScalar failvar  = ftmp[3];
  TacsScalar failstd  = sqrt(failvar);
 
  // Store the evaluated function values
  this->fvals[0] = massmean + massstd;
  this->fvals[1] = failmean + this->beta*failstd;

  printf("mass expectation and std deviation are : %15.10e %15.10e %15.10e\n", massmean, massstd, this->fvals[0]);
  printf("fail expectation and std deviation are : %15.10e %15.10e %15.10e\n", failmean, failstd, this->fvals[1]);

  //----------------------------------------------------------------//
  // setup obj function deriv
  //----------------------------------------------------------------//

  // Do Adjoint solve in TACS
  integrator->integrateAdjoint();

  TACSBVec *dfdx1, *dfdx2;
  integrator->getGradient(0, &dfdx1);
  integrator->getGradient(1, &dfdx2);

  // Compute the derivative of standard deviation 
  dfdx2->axpy(-2.0*massmean, dfdx1);
  if (abs(massvar) > 1.0e-8){
    dfdx2->scale(1.0/(2.0*massstd));
  } else {
    printf("small variance of mass %e \n", massvar);
    dfdx2->scale(0.0);
  }

  // Acess mean and std dev derivatives
  TacsScalar *massmeanderiv, *massstdderiv;
  dfdx1->getArray(&massmeanderiv);
  dfdx2->getArray(&massstdderiv);

  // Objective function gradient
  for (int i = 0; i < n; i++){
    this->dfdx[i] = massmeanderiv[i] + massstdderiv[i];
  }

  //----------------------------------------------------------------//
  // setup constraint function deriv
  //----------------------------------------------------------------//
    
  TACSBVec *dfdx3, *dfdx4;
  integrator->getGradient(2, &dfdx3);
  integrator->getGradient(3, &dfdx4);

  // Compute the derivative of standard deviation 
  dfdx4->axpy(-2.0*failmean, dfdx3);
  if (abs(failvar) > 1.0e-8){
    dfdx4->scale(1.0/(2.0*failstd));
  } else {
    printf("small variance of failure %e \n", failvar);
    dfdx4->scale(0.0);
  }

  // Acess mean and std dev derivatives
  TacsScalar *failmeanderiv, *failstdderiv;
  dfdx3->getArray(&failmeanderiv);
  dfdx4->getArray(&failstdderiv);

  // constraint function gradient
  for (int i = 0; i < n; i++){
    this->dgdx[i] = failmeanderiv[i] + this->beta*failstdderiv[i];
  }

}

bool ProjectionOUU::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                                 Index& nnz_h_lag, IndexStyleEnum& index_style)
{
  // The problem described in ProjectionOUU.hpp has 2 variables, x1, & x2,
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

bool ProjectionOUU::get_bounds_info(Index n, Number* x_l, Number* x_u,
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

bool ProjectionOUU::get_starting_point(Index n, bool init_x, Number* x,
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

bool ProjectionOUU::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{

  if (new_x){
    this->evaluateFuncGrad(n, x);
  }
  
  obj_value = this->fvals[0];
  
  return true;
}

bool ProjectionOUU::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{

  if (new_x){
    this->evaluateFuncGrad(n, x);
  }

  for (int i = 0; i < n; i++){  
    grad_f[i] = this->dfdx[i];
  }

  return true;
}

bool ProjectionOUU::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
  
  if (new_x){
    this->evaluateFuncGrad(n, x);
  }

  g[0] = this->fvals[1];

  return true;
}

bool ProjectionOUU::eval_jac_g(Index n, const Number* x, bool new_x,
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

bool ProjectionOUU::eval_h(Index n, const Number* x, bool new_x,
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

void ProjectionOUU::finalize_solution(SolverReturn status,
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
  int num_steps = 1200;
  double abstol = 1.0e-7;
  double reltol = 1.0e-12;
  SmartPtr<TNLP> mynlp = new ProjectionOUU(nA, nB, nC, tf, num_steps, 
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
