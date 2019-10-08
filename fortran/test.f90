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
  real(8) :: y(2)

  y(1) = 1.0d0
  y(2) = 1.1d0

  z = 1.1d0  

  p1 = normal_parameter(1, -4.0d0, 0.5d0)
  p2 = uniform_parameter(2, -5.0d0, 4.0d0)
  p3 = exponential_parameter(3, 6.0d0, 1.0d0)
  p4 = exponential_parameter(4, 6.0d0, 1.0d0)
  p5 = normal_parameter(5, -4.0d0, 0.5d0)

  pmax = [3,3,4,4,8]

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
    real(8) :: z(5) 
    type(parameter_container) :: pc

    z = [1.01d0, 1.00d0, 1.0001d0, 2.0d0, 0.231312d0]

    call pc % add(p1,pmax(1))
    call pc % add(p2,pmax(2))
    call pc % add(p3,pmax(3))
    call pc % add(p4,pmax(4))
    call pc % add(p5,pmax(5))

    call pc % initialize()
    do j = 1, 100
       do k = 1, pc % get_num_basis_terms()
          psikz = pc % basis(k, z)
          !print *, k, psikz
       end do
    end do

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
