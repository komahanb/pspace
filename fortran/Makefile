default:
	f2py -m forthogonal_polynomials -h orthogonal_polynomials.pyf orthogonal_polynomials.f90 --overwrite-signature
	f2py -m fgaussian_quadrature -h gaussian_quadrature.pyf gaussian_quadrature.f90 --overwrite-signature
	f2py -m fbasis_helper -h basis_helper.pyf basis_helper.f90 --overwrite-signature

python:
	f2py -m forthogonal_polynomials -c orthogonal_polynomials.pyf orthogonal_polynomials.f90
	f2py -m fgaussian_quadrature -c gaussian_quadrature.pyf gaussian_quadrature.f90
	f2py -m fbasis_helper -c basis_helper.pyf basis_helper.f90

fortran:
	caf -O3 -fbounds-check -c orthogonal_polynomials.f90
	caf -O3 -fbounds-check -c gaussian_quadrature.f90
	caf -O3 -fbounds-check -c abstract_parameter_class.f90
	caf -O3 -fbounds-check -c normal_parameter_class.f90
	caf -O3 -fbounds-check -c uniform_parameter_class.f90
	caf -O3 -fbounds-check -c exponential_parameter_class.f90
	caf -O3 -fbounds-check -c class_list.f90
	caf -O3 -fbounds-check -c basis_helper.f90
	caf -O3 -fbounds-check -c quadrature_helper.f90
	caf -O3 -fbounds-check -c parameter_container.f90
	caf -O3 -fbounds-check -c test.f90
	caf -O3 -fbounds-check orthogonal_polynomials.o gaussian_quadrature.o \
		abstract_parameter_class.o normal_parameter_class.o \
		uniform_parameter_class.o exponential_parameter_class.o \
		class_list.o basis_helper.o quadrature_helper.o \
		parameter_container.o test.o -o a.out

clean:
	rm *.so *.pyf *.mod *.o *~ a.out
