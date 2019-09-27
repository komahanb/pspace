cdef extern from "OrthogonalPolynomials.h":
    cdef cppclass OrthogonalPolynomials:
        OrthogonalPolynomials()        
        double hermite(double, int)
        double unit_hermite(double, int)        
        double legendre(double, int)
        double unit_legendre(double, int)
        double laguerre(double, int)
        double unit_laguerre(double, int)
        
cdef class PyOrthogonalPolynomials:
    cdef OrthogonalPolynomials *ptr
    def __cinit__(self):
        self.ptr = new OrthogonalPolynomials()
    def __dealloc__(self):
        del self.ptr
    def hermite(self, z, d):
        return self.ptr.hermite(z, d)
    def unit_hermite(self, z, d):
        return self.ptr.unit_hermite(z, d)
    def laguerre(self, z, d):
        return self.ptr.laguerre(z, d)
    def unit_laguerre(self, z, d):
        return self.ptr.unit_laguerre(z, d)
    def legendre(self, z, d):
        return self.ptr.legendre(z, d)
    def unit_legendre(self, z, d):
        return self.ptr.unit_legendre(z, d)
