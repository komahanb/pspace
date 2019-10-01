module exponential_parameter_class

  use orthogonal_polynomials   , only : laguerre, unit_laguerre
  use gaussian_quadrature      , only : laguerre_quadrature
  use abstract_parameter_class , only : abstract_parameter
  
  implicit none

  ! Exponential parameter type
  type, extends(abstract_parameter) :: exponential_parameter
     real(8) :: mu
     real(8) :: beta
   contains
     procedure :: basis
     procedure :: quadrature
  end type exponential_parameter

  ! Constructor interface for list
  interface exponential_parameter
     module procedure create_exponential_parameter
  end interface exponential_parameter

contains

  !===================================================================!
  ! Constructor for exponential parameter
  !===================================================================!
  
  pure type(exponential_parameter) function create_exponential_parameter(pid, mu, beta) &
       & result(this)

    integer, intent(in) :: pid    
    real(8), intent(in) :: mu
    real(8), intent(in) :: beta
    
    call this % set_parameter_id(pid)
    this % mu = mu
    this % beta = beta

  end function create_exponential_parameter

  !===================================================================!
  ! Evaluate the basis function and return the value
  !===================================================================!
  
  pure real(8) function basis(this, z, d)

    class(exponential_parameter) , intent(in) :: this
    real(8)                      , intent(in) :: z
    integer                      , intent(in) :: d

    basis = unit_laguerre(z,d)

  end function basis

  !===================================================================!
  ! Return the quadrature points and weights
  !===================================================================!

  pure subroutine quadrature(this, npoints, z, y, w)
    
    class(exponential_parameter) , intent(in)    :: this
    integer                      , intent(in)    :: npoints
    real(8)                      , intent(inout) :: z(:), y(:)
    real(8)                      , intent(inout) :: w(:)

    call laguerre_quadrature(npoints, this % mu, this % beta, z, y, w)

  end subroutine quadrature
  
end module exponential_parameter_class
!!$
!!$program test_parameters
!!$  
!!$  use exponential_parameter_class
!!$
!!$  type(exponential_parameter) :: m
!!$  m = exponential_parameter(pid = 1, mu=1.0d0, beta=1.0d0)
!!$  print *, m % mu
!!$  print *, m % beta
!!$
!!$end program test_parameters
