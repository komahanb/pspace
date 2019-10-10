default:
	g++ -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c OrthogonalPolynomials.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c GaussianQuadrature.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c AbstractParameter.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c NormalParameter.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c UniformParameter.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c ExponentialParameter.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check -c QuadratureHelper.cpp
	g++ -Wall -fPIC -O3 -funroll-loops -fbounds-check -fstack-protector-all  --stack-check OrthogonalPolynomials.o \
	GaussianQuadrature.o AbstractParameter.o NormalParameter.o UniformParameter.o \
	ExponentialParameter.o QuadratureHelper.o -o a.out

python:
	python setup.py build_ext --inplace

clean:
	rm *.so *.pyf *.mod *.o *~ a.out
