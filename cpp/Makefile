CFLAGS = -Wall -fPIC -g -DUSE_COMPLEX

default:
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c ArrayList.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c OrthogonalPolynomials.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c GaussianQuadrature.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c AbstractParameter.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c NormalParameter.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c UniformParameter.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c ExponentialParameter.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c ParameterFactory.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c BasisHelper.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c QuadratureHelper.cpp
	mpicxx ${CFLAGS} -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c ParameterContainer.cpp

	# Create dynamic library
	ar rcs libpspace.a  ArrayList.o OrthogonalPolynomials.o \
	GaussianQuadrature.o AbstractParameter.o NormalParameter.o UniformParameter.o \
	ExponentialParameter.o ParameterFactory.o \
	BasisHelper.o QuadratureHelper.o \
	ParameterContainer.o 

	# Create shared object
	mpicxx -shared -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all --stack-check \
	ArrayList.o OrthogonalPolynomials.o \
	GaussianQuadrature.o AbstractParameter.o NormalParameter.o UniformParameter.o \
	ExponentialParameter.o ParameterFactory.o \
	BasisHelper.o QuadratureHelper.o \
	ParameterContainer.o -o libpspace.so

	# Create executable
	mpicxx -I. -L. ${CFLAGS} main.cpp -o a.out -lpspace

python:
	python setup.py build_ext --inplace

clean:
	rm *.so *.pyf *.mod *.o *~ a.out *.gch *.a *.so
