module orthogonal_polynomials

  implicit none

contains
  
  !===================================================================!
  ! Compute the factorial of given integer number
  !===================================================================!
  
  pure real(8) function factorial(n)

    integer, intent(in) :: n
    integer :: i

    factorial = 1.0d0
    do i = 1, n
       factorial = factorial*dble(i)
    end do

  end function factorial

  !===================================================================!
  ! Hermite polynomial of degree d, evaluated at z: H(z,d)
  !===================================================================!
  
  pure real(8) recursive function hermite(z, d) result(hval)

    real(8), intent(in) :: z
    integer, intent(in) :: d

    ! Use two term recursion formulae
    if (d == 0) then
       hval =  1.0d0
    else if (d == 1) then
       hval = z
    else 
       hval = z*hermite(z,d-1) - dble(d-1)*hermite(z,d-2)
    end if

  end function hermite

  !===================================================================!
  ! Normalize the Hermite polynomial \hat{H}(z,d)
  !===================================================================!
  
  pure real(8) function unit_hermite(z,d)

    real(8), intent(in) :: z
    integer, intent(in) :: d

    unit_hermite = hermite(z,d)/sqrt(factorial(d))

  end function unit_hermite

end module orthogonal_polynomials

program test_orthonormal_polynomials

  use orthogonal_polynomials

  implicit none

  integer :: d, i
  real(8), parameter :: z = 1.2d0

  print *, "hello world"

  do i = 0, 10
     print *, i, factorial(i)
  end do

  do i = 0, 4
     print *, unit_hermite(z=z,d=i), hermite(z=z,d=i)/sqrt(factorial(i))
  end do

end program test_orthonormal_polynomials
