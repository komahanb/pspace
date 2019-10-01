module uniform_parameter_class

  use orthogonal_polynomials   , only : legendre, unit_legendre
  use gaussian_quadrature      , only : legendre_quadrature 
  use abstract_parameter_class , only : abstract_parameter
 
  implicit none

  ! Uniform parameter type
  type, extends(abstract_parameter) :: uniform_parameter
     real(8) :: a
     real(8) :: b
   contains
     procedure :: basis
     procedure :: quadrature
  end type uniform_parameter

  ! Constructor interface for list
  interface uniform_parameter
     module procedure create_uniform_parameter
  end interface uniform_parameter

contains

  !===================================================================!
  ! Constructor for uniform parameter
  !===================================================================!
  
  pure type(uniform_parameter) function create_uniform_parameter(pid, a, b) &
       & result(this)

    integer, intent(in) :: pid
    real(8), intent(in) :: a
    real(8), intent(in) :: b

    call this % set_parameter_id(pid)
    this % a = a
    this % b = b

  end function create_uniform_parameter

  !===================================================================!
  ! Evaluate the basis function and return the value
  !===================================================================!
  
  pure real(8) function basis(this, z, d)

    class(uniform_parameter), intent(in) :: this
    real(8)                 , intent(in) :: z
    integer                 , intent(in) :: d

    basis = unit_legendre(z,d)

  end function basis

  !===================================================================!
  ! Return the quadrature points and weights
  !===================================================================!

  pure subroutine quadrature(this, npoints, z, y, w)
    
    class(uniform_parameter), intent(in)    :: this
    integer                 , intent(in)    :: npoints
    real(8)                 , intent(inout) :: z(:), y(:)
    real(8)                 , intent(inout) :: w(:)

    call legendre_quadrature(npoints, this % a, this % b, z, y, w)

  end subroutine quadrature
  
end module uniform_parameter_class

!!$program test_parameters
!!$  
!!$  use uniform_parameter_class
!!$
!!$  type(uniform_parameter) :: m
!!$  m = uniform_parameter(pid = 1, a=1.0d0, b=1.0d0)
!!$  print *, m % a
!!$  print *, m % b
!!$
!!$end program test_parameters
