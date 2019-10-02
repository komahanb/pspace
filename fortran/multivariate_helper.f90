module multivariate_helper

  implicit none

  private
  public :: basis_degrees

contains
  
  subroutine basis_degrees(pmax, indx)

    integer, intent(in)               :: pmax(:)    
    integer, allocatable, intent(out) :: indx(:,:)        

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

  end subroutine basis_degrees

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

end module multivariate_helper

program test_basis

  use multivariate_helper

  integer, allocatable :: idx1(:,:), idx2(:,:), idx3(:,:), idx4(:,:), idx5(:,:)
  integer, parameter   :: pmax1(1) = [2]
  integer, parameter   :: pmax2(2) = [3,2]
  integer, parameter   :: pmax3(3) = [3,2,3]
  integer, parameter   :: pmax4(4) = [3,2,3,3]
  integer, parameter   :: pmax5(5) = [1,2,3,3,2]
  integer :: i

  call basis_degrees(pmax1, idx1)
  do i = 1, size(idx1,dim=2)
     print *, idx1(:,i)
  end do

  call basis_degrees(pmax2, idx2)
  do i = 1, size(idx2,dim=2)
     print *, idx2(:,i)
  end do

  call basis_degrees(pmax3, idx3)
  do i = 1, size(idx3,dim=2)
     print *, idx3(:,i)
  end do

  call basis_degrees(pmax4, idx4)
  do i = 1, size(idx4,dim=2)
     print *, idx4(:,i)
  end do

  call basis_degrees(pmax5, idx5)
  do i = 1, size(idx5,dim=2)
     print *, idx5(:,i)
  end do

end program test_basis
