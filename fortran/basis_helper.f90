module basis_helper

  implicit none

  private
  public :: basis_degrees, sparse !, num_quad_points

contains
  
  pure function sparse(dmapi, dmapj, dmapf) result(filter)

    integer, intent(in) :: dmapi(:), dmapj(:), dmapf(:)
    logical, allocatable :: filter(:)

    integer :: nvars
    integer :: i

    nvars = size(dmapi)
    allocate(filter(nvars))

    do i = 1, nvars
       if (abs(dmapi(i) - dmapj(i)) .le. dmapf(i)) then
          filter(i) = .true.
       else
          filter(i) = .false.
       end if
    end do

  end function sparse

  function basis_degrees(pmax) result(indx)

    integer, intent(in)               :: pmax(:)    
    !integer, allocatable, intent(out) :: indx(:,:)        

    integer, allocatable :: indx(:,:)        

    integer :: nvars
    nvars = size(pmax)

    if (nvars.eq.1) then
       call univariate_degree_index(pmax, indx)
    else if (nvars.eq.2) then
       call bivariate_degree_index(pmax, indx)
    else if (nvars.eq.3) then
       call trivariate_degree_index(pmax, indx)
    else if (nvars.eq.4) then
       call quadvariate_degree_index(pmax, indx)
    else if (nvars.eq.5) then
       call pentavariate_degree_index(pmax, indx)
    else 
       print *, "implement multivariate basis for num vars = ", nvars
       stop
    end if

  end function basis_degrees

  subroutine univariate_degree_index(pmax, indx)
    
    integer, intent(in) :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    

    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii

    nvars = size(pmax)
    nterms = product(1 + pmax)

    allocate(indx(nvars,nterms))
    
    ctr = 0
    do ii = 0, pmax(1)
       ctr = ctr + 1
       indx(:,ctr) = [ii]
    end do

  end subroutine univariate_degree_index
  
  subroutine bivariate_degree_index(pmax, indx)

    integer, intent(in) :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    

    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii, jj

    nvars = size(pmax)
    nterms = product(1 + pmax)

    allocate(indx(nvars,nterms))

    ctr = 0
    do ii = 0, pmax(1)
       do jj = 0, pmax(2)
          ctr = ctr + 1
          indx(:,ctr) = [ii, jj]
       end do
    end do

  end subroutine bivariate_degree_index

  subroutine trivariate_degree_index(pmax, indx)

    integer, intent(in) :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    

    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii, jj, kk

    nvars = size(pmax)
    nterms = product(1 + pmax)

    allocate(indx(nvars,nterms))

    ctr = 0
    do ii = 0, pmax(1)
       do jj = 0, pmax(2)
          do kk = 0, pmax(3)
             ctr = ctr + 1
             indx(:,ctr) = [ii, jj, kk]
          end do
       end do
    end do

  end subroutine trivariate_degree_index

  subroutine quadvariate_degree_index(pmax, indx)

    integer, intent(in) :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    

    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii, jj, kk, ll

    nvars = size(pmax)
    nterms = product(1 + pmax)

    allocate(indx(nvars,nterms))

    ctr = 0
    do ii = 0, pmax(1)
       do jj = 0, pmax(2)
          do kk = 0, pmax(3)
             do ll = 0, pmax(4)
                ctr = ctr + 1
                indx(:,ctr) = [ii, jj, kk, ll]
             end do
          end do
       end do
    end do

  end subroutine quadvariate_degree_index

  subroutine pentavariate_degree_index(pmax, indx)

    integer, intent(in) :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    

    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii, jj, kk, ll, mm

    nvars = size(pmax)
    nterms = product(1 + pmax)

    allocate(indx(nvars,nterms))

    ctr = 0
    do ii = 0, pmax(1)
       do jj = 0, pmax(2)
          do kk = 0, pmax(3)
             do ll = 0, pmax(4)
                do mm = 0, pmax(5)
                   ctr = ctr + 1
                   indx(:,ctr) = [ii, jj, kk, ll, mm]
                end do
             end do
          end do
       end do
    end do

  end subroutine pentavariate_degree_index

end module basis_helper

program test_basis

  use basis_helper

  integer, allocatable :: idx1(:,:), idx2(:,:), idx3(:,:), idx4(:,:), idx5(:,:)
  integer, parameter   :: pmax1(1) = [2]
  integer, parameter   :: pmax2(2) = [3,2]
  integer, parameter   :: pmax3(3) = [3,2,3]
  integer, parameter   :: pmax4(4) = [3,2,3,3]
  integer, parameter   :: pmax5(5) = [1,2,3,3,2]
  integer :: i
  logical, allocatable :: filter(:)

  idx1 = basis_degrees(pmax1)
  do i = 1, size(idx1,dim=2)
     print *, idx1(:,i)
  end do

  idx2 = basis_degrees(pmax2)
  do i = 1, size(idx2,dim=2)
     print *, idx2(:,i)
  end do

  idx3 = basis_degrees(pmax3)
  do i = 1, size(idx3,dim=2)
     print *, idx3(:,i)
  end do

  idx4 = basis_degrees(pmax4)
  do i = 1, size(idx4,dim=2)
     print *, idx4(:,i)
  end do

  idx5 = basis_degrees(pmax5)
  do i = 1, size(idx5,dim=2)
     print *, idx5(:,i)
  end do

  filter = sparse([0,0], [0,2], [1,1])
  print *, filter

end program test_basis
