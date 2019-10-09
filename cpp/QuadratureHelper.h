class QuadratureHelper {
 public:
  // Constructor and destructor
  QuadratureHelper();
  ~QuadratureHelper();

  // find tensor product of 1d rules
  void tensorProduct(const int nvars,
                     const int *nqpts,
                     const int **zp, const int **yp, const int **wp,
                     int **zz, int **yy, int **ww);
};


