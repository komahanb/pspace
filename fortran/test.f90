program main

  use normal_parameter_class
  use uniform_parameter_class
  use exponential_parameter_class

  type(normal_parameter) :: p1
  type(uniform_parameter) :: p2
  type(exponential_parameter) :: p3

  real(8) :: z

  real(8) :: y(2)

  y(1) = 1.0d0
  y(2) = 1.1d0
  
  z = 1.1d0  

  p1 = normal_parameter(1,1.0d0,1.0d0)
  p2 = uniform_parameter(2,1.0d0,2.0d0)
  p3 = exponential_parameter(3,1.0d0,1.0d0)

  print *, p1 % basis(y, 1), p2 % basis(y, 1), p3 % basis(y, 1)
  
  do i = 0, 15
     print *, p1 % basis(z, i), p2 % basis(z, i), p3 % basis(z, i)
  end do

end program main
