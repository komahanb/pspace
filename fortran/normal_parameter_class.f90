module normal_parameter_class

  use orthogonal_polynomials, only : hermite, unit_hermite
  use gaussian_quadrature   , only : hermite_quadrature
  
  implicit none

  ! Normal parameter type
  type :: normal_parameter
     real(8) :: mu
     real(8) :: sigma
   contains
     procedure :: basis
     procedure :: quadrature
  end type normal_parameter

  ! Constructor interface for list
  interface normal_parameter
     module procedure create_normal_parameter
  end interface normal_parameter

contains

  !===================================================================!
  ! Constructor for normal parameter
  !===================================================================!
  
  pure type(normal_parameter) function create_normal_parameter(mu, sigma) &
       & result(this)

    real(8), intent(in) :: mu
    real(8), intent(in) :: sigma
    
    this % mu = mu
    this % sigma = sigma

  end function create_normal_parameter

  !===================================================================!
  ! Evaluate the basis function and return the value
  !===================================================================!
  
  pure real(8) function basis(this, z, d)

    class(normal_parameter) , intent(in) :: this
    real(8)                 , intent(in) :: z
    integer                 , intent(in) :: d

    basis = unit_hermite(z,d)

  end function basis

  !===================================================================!
  ! Return the quadrature points and weights
  !===================================================================!

  pure subroutine quadrature(this, npoints, z, y, w)
    
    class(normal_parameter) , intent(in)    :: this
    integer                 , intent(in)    :: npoints
    real(8)                 , intent(inout) :: z(:), y(:)
    real(8)                 , intent(inout) :: w(:)

    call hermite_quadrature(npoints, this % mu, this % sigma, z, y, w)

  end subroutine quadrature
  
end module normal_parameter_class

program test_parameters

  use normal_parameter_class

  type(normal_parameter) :: m

  m = normal_parameter(mu=1.0d0, sigma=1.0d0)

  print *, m % mu
  print *, m % sigma

end program test_parameters
