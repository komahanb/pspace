module class_parameter

  use orthogonal_polynomials, only : hermite, unit_hermite
  use gaussian_quadrature   , only : hermite_quadrature
  
  implicit none

  ! Normal parameter type
  type :: normal
     real(8) :: mu
     real(8) :: sigma
   contains
     procedure :: basis
     procedure :: quadrature
  end type normal

  ! Constructor interface for list
  interface normal
     module procedure create_normal
  end interface normal

contains

  !===================================================================!
  ! Constructor for normal parameter
  !===================================================================!
  
  pure type(normal) function create_normal(mu, sigma) &
       & result(this)

    real(8), intent(in) :: mu
    real(8), intent(in) :: sigma
    
    this % mu = mu
    this % sigma = sigma

  end function create_normal

  !===================================================================!
  ! Evaluate the basis function and return the value
  !===================================================================!
  
  pure real(8) function basis(this, z, d)

    class(normal) , intent(in) :: this
    real(8)       , intent(in) :: z
    integer       , intent(in) :: d

    basis = unit_hermite(z,d)

  end function basis

  !===================================================================!
  ! Return the quadrature points and weights
  !===================================================================!

  pure subroutine quadrature(this, npoints, z, y, w)
    
    class(normal) , intent(in)    :: this
    integer       , intent(in)    :: npoints
    real(8)       , intent(inout) :: z(:), y(:)
    real(8)       , intent(inout) :: w(:)

    call hermite_quadrature(npoints, this % mu, this % sigma, z, y, w)

  end subroutine quadrature

end module class_parameter

program test_parameters

  use class_parameter

  type(normal) :: m
  m = normal(mu=1.0d0, sigma=1.0d0)
  print *, m % mu
  print *, m % sigma

end program test_parameters
