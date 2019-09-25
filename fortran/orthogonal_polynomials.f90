module orthogonal_polynomials

  implicit none

  private
  public :: hermite, unit_hermite

contains

  !===================================================================!
  ! Compute the factorial of given integer number
  !===================================================================!

  pure function factorial(n)

    integer, intent(in) :: n
    real(8) :: factorial

    integer :: i

    factorial = 1.0d0
    do i = 1, n
       factorial = factorial*dble(i)
    end do

  end function factorial

  !===================================================================!
  ! Hermite polynomial of degree d, evaluated at z: H(z,d)
  !===================================================================!

  pure recursive function recursive_hermite(z, d) result(hval)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: hval

    ! Use two term recursion formulae
    if (d == 0) then
       hval =  1.0d0
    else if (d == 1) then
       hval = z
    else 
       hval = z*recursive_hermite(z,d-1) - dble(d-1)*recursive_hermite(z,d-2)
    end if

  end function recursive_hermite

  pure function explicit_hermite(z, d) result(hval)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: hval

    ! Use two term recursion formulae
    if (d .eq. 0) then
       hval =  1.0d0
    else if (d .eq. 1) then
       hval = z
    else if (d .eq. 2) then
       hval = z**2 - 1.0d0
    else if (d .eq. 3) then
       hval = z**3 - 3.0d0*z
    else if (d .eq. 4) then
       hval = z**4 - 6.0d0*z*z + 3.0d0
    end if

  end function explicit_hermite

  pure function hermite(z, d)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: hermite

    if (d .le. 4) then 
       hermite = explicit_hermite(z, d)
    else
       hermite = recursive_hermite(z, d)
    end if

  end function hermite

  !===================================================================!
  ! Normalize the Hermite polynomial \hat{H}(z,d)
  !===================================================================!

  pure function unit_hermite(z,d)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: unit_hermite

    unit_hermite = hermite(z,d)/sqrt(factorial(d))

  end function unit_hermite

end module orthogonal_polynomials

program test_orthonormal_polynomials

  use orthogonal_polynomials
  implicit none

  integer :: d, i
  real(8), parameter :: z = 1.1d0

  print *, "hello world"

  do i = 0, 44
     print *, hermite(z,i) !unit_hermite(z=z,d=i), hermite(z=z,d=i)/sqrt(factorial(i))
  end do

end program test_orthonormal_polynomials
