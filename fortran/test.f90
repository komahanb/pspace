program main

  use normal_parameter_class
  use uniform_parameter_class
  use exponential_parameter_class

  use basis_helper, only : basis_degrees

  integer, parameter :: nvars = 5
  integer            :: pmax(nvars)

  type(normal_parameter)      :: p1, p5
  type(uniform_parameter)     :: p2
  type(exponential_parameter) :: p3
  type(exponential_parameter) :: p4

  integer, allocatable :: vardeg(:,:)
  integer :: i

  real(8) :: z
!!$  real(8) :: y(2)
  
  real(8), allocatable, dimension(:) :: z1, z2
  real(8), allocatable, dimension(:) :: y1, y2
  real(8), allocatable, dimension(:) :: w1, w2
!!$  integer :: nqpts(2)

!!$  nqpts = [3,4]
!!$  
!!$  y(1) = 1.0d0
!!$  y(2) = 1.1d0
!!$
!!$  z = 1.1d0  

  p1 = normal_parameter(1, -4.0d0, 0.5d0)
  p2 = uniform_parameter(2, -5.0d0, 4.0d0)
  p3 = exponential_parameter(3, 6.0d0, 1.0d0)
  p4 = exponential_parameter(4, 6.0d0, 1.0d0)
  p5 = normal_parameter(5, -4.0d0, 0.5d0)

  pmax = [3,3,4,4,2]

!!$  allocate(z1(nqpts(1)),y1(nqpts(1)),w1(nqpts(1)))
!!$  allocate(z2(nqpts(2)),y2(nqpts(2)),w2(nqpts(2)))
!!$  
!!$  call p1 % quadrature(nqpts(1), z1, y1, w1)
!!$  call p2 % quadrature(nqpts(2), z2, y2, w2)
!!$
!!$  call pc % quadrature(q, zq, yq, wq)
!!$
!!$  print *, z1, y1, w1
!!$  print *, z2, y2, w2
  
!!$  vardeg = basis_degrees(pmax)
!!$  do i = 1, size(vardeg,dim=2)
!!$     print *, i, vardeg(:,i)
!!$  end do

!!$  print *, p1 % get_parameter_id()
!!$  print *, p2 % get_parameter_id()
!!$  print *, p3 % get_parameter_id()
  
!!$
!!$  stop
!!$
!!$  do i = 1, size(vardeg,dim=2)
!!$     do j = 1, 3
!!$        print *, p1 % basis(z,vardeg(j,i))
!!$     end do
!!$  end do
!!$
  
  test : block

    use parameter_container_class, only : parameter_container

    integer :: k, num_terms
    real(8) :: psikz
    real(8) :: psiz
    real(8) :: z(nvars) 
    type(parameter_container) :: pc

    integer :: nqpts(nvars), q, totnqpts
    integer :: vmaxdeg(nvars)
    real(8) :: zq(nvars), yq(nvars), wq


    z = [1.01d0, 1.00d0, 1.0001d0, 2.0d0, 0.231312d0]
    !z = [1.01d0, 1.00d0, 1.0001d0, 2.0d0] 

    call pc % add(p1)
    call pc % add(p2)
    call pc % add(p3)
    call pc % add(p4)
    call pc % add(p5)

    call pc % initialize_basis(pmax)

!!$    do j = 1, 1
!!$       do k = 1, pc % get_num_basis_terms()
!!$          psikz = pc % basis(k, z)
!!$          print *, k, psikz
!!$       end do
!!$    end do
!!$    
!!$    !stop

    ! nqpts = [1, 1, 1, 1, 3]
    ! nqpts = [3, 3, 2, 2] !, 3]

!!$        
!!$    call pc % initialize_quadrature(nqpts)
!!$    totnqpts = pc % get_num_quadrature_points()
!!$    do q = 1, totnqpts
!!$       call pc % get_quadrature(q, zq, yq, wq)
!!$       print *, q, "zq", zq, "w=", wq
!!$    end do

    ! initialize quadrature with num-quad-points for each variable

    ! nqpts = nqpts_from_max_degree(pmax)
    call pc % initialize_quadrature(1 + pmax)
       
    do k = 1, pc % get_num_basis_terms()

       do q = 1, pc % get_num_quadrature_points()

          call pc % get_quadrature(q, zq, yq, wq)

          psikz = pc % basis(k, zq)

          !write(*,'(i6, i6, f13.6)') k-1, q-1, psikz

       end do

    end do

    print *,  pc % get_num_basis_terms()*pc % get_num_quadrature_points()

    ! for every basis term
    !   initialize quadrature based on degree
    !   for every quadrature node
    !     evaluate basis

  end block test
  
!!$    
!!$    type param_list
!!$       class(abstract_parameter), pointer :: p => null()
!!$    end type param_list
!!$
!!$    
!!$    type(param_list), dimension(5) :: plist
!!$
!!$    plist(1) % p => p1
!!$    plist(2) % p => p2
!!$    plist(3) % p => p3
    ! plist(4) % p => a
    
    !type(parameter_container) :: pc

    !pc = parameter_container()

!!    print *, loc(p1), loc(p2)

!!$    select type(p)
!!$    type is (integer)
!!$    class is (abstract_parameter)
!!$    class default
!!$    end select

!!$    print *, plist(1) % p % get_parameter_id()
!!$    print *, plist(2) % p % get_parameter_id()
!!$    print *, plist(3) % p % get_parameter_id()

    
  !  print *, "aa", loc(plist(1) % p)
    !allocate(pc % parameters(4))


    !pc % parameters(1) => p1
  
end program main
