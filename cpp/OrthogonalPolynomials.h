class OrthogonalPolynomials {
 public:

  // Constructor and destructor
  OrthogonalPolynomials();
  ~OrthogonalPolynomials();

  //  Get Hermite polynomials  
  double hermite(double z, int d);

 private:
  // Get factorial of given integer
  double factorial(int n);
};
