program main

  use normal_parameter_class
  use uniform_parameter_class
  use exponential_parameter_class
  use basis_helper, only : basis_degrees

  integer, parameter :: nvars = 3
  integer            :: pmax(nvars)

  type(normal_parameter)      :: p1
  type(uniform_parameter)     :: p2
  type(exponential_parameter) :: p3

  integer, allocatable :: vardeg(:,:)
  integer :: i

  real(8) :: z
  real(8) :: y(2)

  y(1) = 1.0d0
  y(2) = 1.1d0

  z = 1.1d0  

  p1 = normal_parameter(1,1.0d0,1.0d0)
  p2 = uniform_parameter(2,1.0d0,2.0d0)
  p3 = exponential_parameter(3,1.0d0,1.0d0)

  pmax = [2,4,2]
  vardeg = basis_degrees(pmax)
  do i = 1, size(vardeg,dim=2)
     print *, i, vardeg(:,i)
  end do

  print *, p1 % get_parameter_id()
  print *, p2 % get_parameter_id()
  print *, p3 % get_parameter_id()
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
    
    type(parameter_container) :: pc
    real(8) :: psiz
    
    call pc % add(p1)
    call pc % add(p2)
    call pc % add(p3)

    psiz = pc % basis([1.0d0, 1.1d0, 1.2d0], [0,0,0])

    print *, psiz
    
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
    
    
  end block test

end program main
