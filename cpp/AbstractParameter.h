class AbstractParameter {
 public:
  // Constructor and destructor
  AbstractParameter();
  ~AbstractParameter();

  // Deferred procedures
  virtual void quadrature(int npoints, double *z, double *y, double *w) = 0;
  virtual void basis(double *z, int d) = 0;

  // Implemented procedures
  int getParameterID();  
  void setParameterID(int pid);
  
 private:
  int parameter_id;
};

/*
  Constructor
*/
AbstractParameter::AbstractParameter(){}

/*
  Destructor
*/
AbstractParameter::~AbstractParameter(){}

/*
  Setter for parameter ID
*/
void AbstractParameter::setParameterID(int pid){
  this->parameter_id = pid;
}

/*
  Getter for parameter ID
*/
int AbstractParameter::getParameterID(){
  return this->parameter_id;
}

