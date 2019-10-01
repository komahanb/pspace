module orthogonal_polynomials

  implicit none

  !private
  !public :: hermite , unit_hermite
  !public :: laguerre, unit_laguerre
  !public :: legendre, unit_legendre

contains

  !===================================================================!
  ! Compute the factorial of given integer number
  !===================================================================!

  pure function factorial(n)

    integer, intent(in) :: n
    real(8) :: factorial
    integer :: i
    
    if ( n .eq. 0 ) then
       factorial = 1.0d0
    else if ( n .eq. 1 ) then
       factorial = 1.0d0
    else if ( n .eq. 2 ) then   
       factorial = 2.0d0       
    else if ( n .eq. 3 ) then
       factorial = 6.0d0              
    else if ( n .eq. 4 ) then
       factorial = 24.0d0                     
    else if ( n .eq. 5 ) then
       factorial = 120.0d0                            
    else if ( n .eq. 6 ) then
       factorial = 720.0d0                                          
    else if ( n .eq. 7 ) then
       factorial = 5040.0d0                                          
    else if ( n .eq. 8 ) then
       factorial = 40320.0d0                                                        
    else if ( n .eq. 9 ) then
       factorial = 362880.0d0
    else if ( n .eq. 10 ) then
       factorial = 3628800.0d0
    else    
       factorial = 1.0d0
       do i = 1, n
          factorial = factorial*dble(i)
       end do
    end if

  end function factorial

  !===================================================================!
  ! Combination nCr = n!/((n-r)!r!)
  !===================================================================!
  
  pure function comb(n,r)

    integer, intent(in) :: n, r
    real(8) :: comb
    real(8) :: nfact, rfact, nrfact
    
    nfact  = factorial(n)
    rfact  = factorial(r)
    nrfact = factorial(n-r)

    comb = nfact/(rfact*nrfact)
    
  end function comb
  
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
       lval = z**2 - 4.0d0*z + 2.0d0
       lval = lval/factorial(2)
    else if (d .eq. 3) then
       lval = -z**3 + 9.0d0*z**2 - 18.0d0*z + 6.0d0
       lval = lval/factorial(3)
    else if (d .eq. 4) then
       lval = z**4 - 16.0d0*z**3 + 72.0d0*z**2 - 96.0d0*z + 24.0d0
       lval = lval/factorial(4)
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

  
  !===================================================================!
  ! Legendre polynomial of degree d, evaluated at z: P(z,d)
  !===================================================================!
  
  pure recursive function general_legendre(z, d) result(pval)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    integer :: k
    real(8) :: pval

    ! using general formula
    pval = 0.0d0
    do k = 0, d
       pval = pval + comb(d,k)*comb(d+k,k)*(-z)**k
    end do
    pval = pval*(-1.0d0)**d       

  end function general_legendre

  pure recursive function explicit_legendre(z, d) result(pval)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: pval

    ! Use two term recursion formulae
    if (d .eq. 0) then
       pval = 1.0d0
    else if (d .eq. 1) then
       pval = 2.0d0*z - 1.0d0
    else if (d .eq. 2) then
       pval = 6.0d0*z**2 - 6.0d0*z + 1
    else if (d .eq. 3) then
       pval = 20.0d0*z**3 -30.0d0*z**2 + 12.0d0*z - 1.0d0
    else if (d .eq. 4) then
       pval = 70.0d0*z**4 - 140.0d0*z**3 + 90.0d0*z**2 - 20.0d0*z + 1.0d0
    end if

  end function explicit_legendre

  pure function legendre(z, d)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: legendre

    if (d .le. 4) then 
       legendre = explicit_legendre(z, d)
    else
       legendre = general_legendre(z, d)
    end if

  end function legendre

  !===================================================================!
  ! Normalize the Legendre polynomial \hat{L}(z,d)
  !===================================================================!

  pure function unit_legendre(z,d)

    real(8), intent(in) :: z
    integer, intent(in) :: d
    real(8) :: unit_legendre

    unit_legendre = legendre(z,d)*sqrt(dble(2*d+1))

  end function unit_legendre
  
end module orthogonal_polynomials

!!$program test_orthonormal_polynomials
!!$
!!$  use orthogonal_polynomials
!!$  implicit none
!!$
!!$  integer :: i, n, j
!!$  real(8), parameter :: z = 1.1d0
!!$  integer, parameter :: max_order = 9, nruns = 100000
!!$  real(8) :: a
!!$
!!$  do n = 1, 10000
!!$     do i = 0, max_order
!!$        a = hermite(z,i)
!!$ !       print *, i, legendre(z,i)
!!$ !       print *, i, laguerre(z,i)             
!!$     end do
!!$  end do
!!$
!!$  print *, a
!!$
!!$  do j = 1, nruns
!!$  print *, "hermite"
!!$  do i = 0, max_order
!!$     write(*,"(i2,F10.3,F10.3)") i, hermite(z,i), unit_hermite(z,i)
!!$  end do
!!$
!!$  print *, ""
!!$  print *, "legendre"
!!$  do i = 0, max_order
!!$     write(*,"(i2,F10.3,F10.3)") i, legendre(z,i), unit_legendre(z,i)
!!$  end do
!!$
!!$  print *, ""
!!$  print *, "laguerre"
!!$  do i = 0, max_order
!!$     write(*,"(i2,F10.3,F10.3)") i, laguerre(z,i), unit_laguerre(z,i)
!!$  end do
!!$  end do
!!$
!!$end program test_orthonormal_polynomials
