default:
	f2py -m forthogonal_polynomials -h orthogonal_polynomials.pyf orthogonal_polynomials.f90 --overwrite-signature
	f2py -m fgaussian_quadrature -h gaussian_quadrature.pyf gaussian_quadrature.f90 --overwrite-signature

python:
	f2py -m forthogonal_polynomials -c orthogonal_polynomials.pyf orthogonal_polynomials.f90
	f2py -m fgaussian_quadrature -c gaussian_quadrature.pyf gaussian_quadrature.f90

clean:
	rm *.so *.pyf *.mod *.o *~
