module orthogonal_polynomials

  implicit none

  !private
  !public :: hermite , unit_hermite
  !public :: laguerre, unit_laguerre

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

  !===================================================================!
  ! Laguerre polynomial of degree d, evaluated at z: L(z,d)
  !===================================================================!
  
  pure recursive function recursive_laguerre(z, d) result(lval)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: lval

    ! Use two term recursion formulae
    if (d == 0) then
       lval = 1.0d0
    else if (d == 1) then
       lval = 1.0d0 - z  
    else 
       lval = ((dble(2*d-1)-z)*recursive_laguerre(z,d-1) &
            & - dble(d-1)*recursive_laguerre(z,d-2))/dble(d)
    end if

  end function recursive_laguerre
 
  pure recursive function explicit_laguerre(z, d) result(lval)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: lval

    ! Use two term recursion formulae
    if (d .eq. 0) then
       lval = 1.0d0
    else if (d .eq. 1) then
       lval = 1.0d0 - z
    else if (d .eq. 2) then
       lval = (z**2 - 4.0d0*z + 2.0d0)/factorial(2)
    else if (d .eq. 3) then
       lval = (-z**3 + 9.0d0*z**2 - 18.0d0*z + 6.0d0)/factorial(3)
    else if (d .eq. 4) then
       lval = (z**4 - 16.0d0*z**3 + 72.0d0*z**2 - 96.0d0*z + 24.0d0)/factorial(4)
    end if
    
  end function explicit_laguerre

  pure function laguerre(z, d)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: laguerre

    if (d .le. 4) then 
       laguerre = explicit_laguerre(z, d)
    else
       laguerre = recursive_laguerre(z, d)
    end if

  end function laguerre

  !===================================================================!
  ! Normalize the Laguerre polynomial \hat{L}(z,d)
  !===================================================================!

  pure function unit_laguerre(z,d)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: unit_laguerre

    unit_laguerre = laguerre(z,d)

  end function unit_laguerre

end module orthogonal_polynomials

program test_orthonormal_polynomials

  use orthogonal_polynomials
  implicit none

  integer :: d, i
  real(8), parameter :: z = 1.1d0

  print *, "hermite"
  do i = 0, 10
     print *, i, hermite(z,i), recursive_hermite(z,i) !unit_hermite(z=z,d=i), hermite(z=z,d=i)/sqrt(factorial(i))
  end do

  print *, "laguerre"
  do i = 0, 10
     print *, i, laguerre(z,i), recursive_laguerre(z,i) !unit_hermite(z=z,d=i), hermite(z=z,d=i)/sqrt(factorial(i))
  end do

  stop

  do i = 0, 44
     print *, hermite(z,i) !unit_hermite(z=z,d=i), hermite(z=z,d=i)/sqrt(factorial(i))
  end do

end program test_orthonormal_polynomials
