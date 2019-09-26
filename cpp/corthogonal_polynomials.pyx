cdef extern from "OrthogonalPolynomials.h":
    cdef cppclass OrthogonalPolynomials:
        Rectangle()
        double hermite(double, int)
        
cdef class PyOrthogonalPolynomials:
    cdef OrthogonalPolynomials *ptr
    def __cinit__(self):
        self.ptr = new OrthogonalPolynomials()
    def __dealloc__(self):
        del self.ptr
    def hermite(self, z, d):
        return self.ptr.hermite(z, d)
