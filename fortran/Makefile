default:
	f2py -m forthogonal_polynomials -h orthogonal_polynomials.pyf orthogonal_polynomials.f90 --overwrite-signature

compile:
	f2py -m forthogonal_polynomials -c orthogonal_polynomials.pyf orthogonal_polynomials.f90

clean:
	rm *.so *.pyf *.mod *.o *~
