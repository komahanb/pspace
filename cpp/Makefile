default:
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c ArrayList.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c OrthogonalPolynomials.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c GaussianQuadrature.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c AbstractParameter.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c NormalParameter.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c UniformParameter.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c ExponentialParameter.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c ParameterFactory.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c BasisHelper.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c QuadratureHelper.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c TACSParameterContainer.cpp
	mpicxx -Wall  -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c ParameterContainer.cpp

	# Create dynamic library
	ar rcs libpspace.a  ArrayList.o OrthogonalPolynomials.o \
	GaussianQuadrature.o AbstractParameter.o NormalParameter.o UniformParameter.o \
	ExponentialParameter.o ParameterFactory.o \
	BasisHelper.o QuadratureHelper.o \
	TACSParameterContainer.o ParameterContainer.o 

	# Create shared object
	mpicxx -shared -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all --stack-check \
	ArrayList.o OrthogonalPolynomials.o \
	GaussianQuadrature.o AbstractParameter.o NormalParameter.o UniformParameter.o \
	ExponentialParameter.o ParameterFactory.o \
	BasisHelper.o QuadratureHelper.o \
	TACSParameterContainer.o ParameterContainer.o -o libpspace.so

	# Create executable
	mpicxx -I. -L. main.cpp -o a.out -lpspace

python:
	python setup.py build_ext --inplace

clean:
	rm *.so *.pyf *.mod *.o *~ a.out *.gch
