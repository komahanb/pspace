module basis_helper

  use class_list, only : list

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

  pure function basis_degrees(pmax) result(indx)

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
!!$    else 
!!$       print *, "implement multivariate basis for num vars = ", nvars
!!$       stop
    end if

  end function basis_degrees

  pure subroutine univariate_degree_index(pmax, indx)

    integer, intent(in)               :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    
    type(list), allocatable           :: degree_list(:)
    integer   , allocatable           :: tmp(:,:)
    
    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii
    integer :: npentries, num_total_degrees

    nvars = size(pmax)
    nterms = product(1 + pmax)
    num_total_degrees = 1 + sum(pmax)
    
    ! Allocate space for degreewise indices
    allocate(degree_list(num_total_degrees))
    do ii = 1, num_total_degrees
       degree_list(ii) = list(nvars,nterms)
    end do

    ! Add indices degreewise
    do ii = 0, pmax(1)
       call degree_list(ii+1) % add_entry([ii])
    end do
    
    ! Flatten list with ascending degrees
    allocate(indx(nvars,nterms))
    ctr = 1
    do ii = 1, num_total_degrees !0, pmax(1)
       npentries = degree_list(ii) % num_entries
       call degree_list(ii) % get_entries(tmp)
       indx(:, ctr:ctr+npentries-1) = tmp(:,:)
       ctr = ctr + npentries
       deallocate(tmp)
    end do

    deallocate(degree_list)

  end subroutine univariate_degree_index
  
  pure subroutine bivariate_degree_index(pmax, indx)

    integer, intent(in)               :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    
    type(list), allocatable           :: degree_list(:)
    integer   , allocatable           :: tmp(:,:)
    
    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii, jj
    integer :: npentries, num_total_degrees

    nvars = size(pmax)
    nterms = product(1 + pmax)

    ! Allocate space for degreewise indices
    num_total_degrees = 1 + sum(pmax)
    allocate(degree_list(num_total_degrees))
    do ii = 1, num_total_degrees
       degree_list(ii) = list(nvars,nterms)
    end do
    
    ! Add indices degreewise
    do ii = 0, pmax(1)
       do jj = 0, pmax(2)
          call degree_list(ii+jj+1) % add_entry([ii,jj])
       end do
    end do
    
    ! Flatten list with ascending degrees
    allocate(indx(nvars,nterms))
    ctr = 1
    do ii = 1, num_total_degrees
       npentries = degree_list(ii) % num_entries
       call degree_list(ii) % get_entries(tmp)
       indx(:, ctr:ctr+npentries-1) = tmp(:,:)
       ctr = ctr + npentries
       deallocate(tmp)
    end do

    deallocate(degree_list)

  end subroutine bivariate_degree_index

  pure subroutine trivariate_degree_index(pmax, indx)

    integer, intent(in)               :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    
    type(list), allocatable           :: degree_list(:)
    integer   , allocatable           :: tmp(:,:)

    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii, jj, kk
    integer :: npentries, num_total_degrees

    nvars = size(pmax)
    nterms = product(1 + pmax)

    ! Allocate space for degreewise indices
    num_total_degrees = 1 + sum(pmax)
    allocate(degree_list(num_total_degrees))
    do ii = 1, num_total_degrees
       degree_list(ii) = list(nvars,nterms)
    end do
    
    ! Add indices degreewise
    do ii = 0, pmax(1)
       do jj = 0, pmax(2)
          do kk = 0, pmax(3)
             call degree_list(ii+jj+kk+1) % add_entry([ii,jj,kk])
          end do
       end do
    end do

    ! Flatten list with ascending degrees
    allocate(indx(nvars,nterms))
    ctr = 1
    do ii = 1, num_total_degrees
       npentries = degree_list(ii) % num_entries
       call degree_list(ii) % get_entries(tmp)
       indx(:, ctr:ctr+npentries-1) = tmp(:,:)
       ctr = ctr + npentries
       deallocate(tmp)
    end do

    deallocate(degree_list)

  end subroutine trivariate_degree_index

  pure subroutine quadvariate_degree_index(pmax, indx)

    integer, intent(in)               :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    
    type(list), allocatable           :: degree_list(:)
    integer   , allocatable           :: tmp(:,:)

    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii, jj, kk, ll
    integer :: npentries, num_total_degrees

    nvars = size(pmax)
    nterms = product(1 + pmax)

    ! Allocate space for degreewise indices
    num_total_degrees = 1 + sum(pmax)
    allocate(degree_list(num_total_degrees))
    do ii = 1, num_total_degrees
       degree_list(ii) = list(nvars,nterms)
    end do
    
    ! Add indices degreewise
    do ii = 0, pmax(1)
       do jj = 0, pmax(2)
          do kk = 0, pmax(3)
             do ll = 0, pmax(4)
                call degree_list(ii+jj+kk+ll+1) % add_entry([ii,jj,kk,ll])
             end do
          end do
       end do
    end do

    ! Flatten list with ascending degrees
    allocate(indx(nvars,nterms))
    ctr = 1
    do ii = 1, num_total_degrees
       npentries = degree_list(ii) % num_entries
       call degree_list(ii) % get_entries(tmp)
       indx(:, ctr:ctr+npentries-1) = tmp(:,:)
       ctr = ctr + npentries
       deallocate(tmp)
    end do

    deallocate(degree_list)

  end subroutine quadvariate_degree_index

  pure subroutine pentavariate_degree_index(pmax, indx)

    integer, intent(in)               :: pmax(:)
    integer, allocatable, intent(out) :: indx(:,:)    
    type(list), allocatable           :: degree_list(:)
    integer   , allocatable           :: tmp(:,:)

    ! locals
    integer :: nvars, nterms, ctr
    integer :: ii, jj, kk, ll, mm
    integer :: npentries, num_total_degrees

    nvars = size(pmax)
    nterms = product(1 + pmax)

    ! Allocate space for degreewise indices
    num_total_degrees = 1 + sum(pmax)
    allocate(degree_list(num_total_degrees))
    do ii = 1, num_total_degrees
       degree_list(ii) = list(nvars,nterms)
    end do
    
    ! Add indices degreewise
    do ii = 0, pmax(1)
       do jj = 0, pmax(2)
          do kk = 0, pmax(3)
             do ll = 0, pmax(4)
                do mm = 0, pmax(5)
                   call degree_list(ii+jj+kk+ll+mm+1) % add_entry([ii,jj,kk,ll,mm])
                end do
             end do
          end do
       end do
    end do

    ! Flatten list with ascending degrees
    allocate(indx(nvars,nterms))
    ctr = 1
    do ii = 1, num_total_degrees
       npentries = degree_list(ii) % num_entries
       call degree_list(ii) % get_entries(tmp)
       indx(:, ctr:ctr+npentries-1) = tmp(:,:)
       ctr = ctr + npentries
       deallocate(tmp)
    end do

    deallocate(degree_list)

  end subroutine pentavariate_degree_index

end module basis_helper

subroutine test_basis

  use basis_helper

  integer, allocatable :: idx1(:,:), idx2(:,:), idx3(:,:), idx4(:,:), idx5(:,:)
  integer, parameter   :: pmax1(1) = [5]
  integer, parameter   :: pmax2(2) = [3,4]
  integer, parameter   :: pmax3(3) = [3,1,3]
  integer, parameter   :: pmax4(4) = [3,2,3,3]
  integer, parameter   :: pmax5(5) = [1,2,3,3,2]

  integer :: i
  logical, allocatable :: filter(:)

  print *, "1 variable"
  idx1 = basis_degrees(pmax1)
  do i = 1, size(idx1,dim=2)
     print *, i, idx1(:,i), sum(idx1(:,i))
  end do
  
  print *, "2 variables"
  idx2 = basis_degrees(pmax2)
  do i = 1, size(idx2,dim=2)
     print *, i, idx2(:,i), sum(idx2(:,i))
  end do

  print *, "3 variables"
  idx3 = basis_degrees(pmax3)
  do i = 1, size(idx3,dim=2)
     print *, i, idx3(:,i), sum(idx3(:,i))
  end do

  print *, "4 variables"
  idx4 = basis_degrees(pmax4)
  do i = 1, size(idx4,dim=2)
     print *, i, idx4(:,i), sum(idx4(:,i))
  end do

  print *, "5 variables"
  idx5 = basis_degrees(pmax5)
  do i = 1, size(idx5,dim=2)
     print *, i, idx5(:,i), sum(idx5(:,i))     
  end do

!!$  stop
!!$  filter = sparse([0,0], [0,2], [1,1])
!!$  print *, filter

end subroutine test_basis
