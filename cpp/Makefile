default:
	g++ -Wall -fPIC -O3 -funroll-loops -c OrthogonalPolynomials.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -c GaussianQuadrature.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -c AbstractParameter.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -c NormalParameter.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -c UniformParameter.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -c ExponentialParameter.cpp
	g++ -Wall -fPIC -O3 -funroll-loops OrthogonalPolynomials.o \
	GaussianQuadrature.o AbstractParameter.o NormalParameter.o UniformParameter.o \
	ExponentialParameter.o -o a.out

python:
	python setup.py build_ext --inplace

clean:
	rm *.so *.pyf *.mod *.o *~ a.out
