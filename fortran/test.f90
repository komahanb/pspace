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

  stop

  do i = 1, size(vardeg,dim=2)
     do j = 1, 3
        print *, p1 % basis(z,vardeg(j,i))
     end do
  end do

end program main
