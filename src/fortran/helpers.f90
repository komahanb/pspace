module helper

contains

  subroutine basis(parameter_degrees)

    integer, intent(in) :: parameter_degrees(:)

    integer :: num_parameters
    integer :: num_basis_terms

    num_basis_terms = product(parameter_degrees)

    print *, num_basis_terms
    
  end subroutine basis

end module helper

program main

  use helper
  
  integer, parameter :: parameter_degrees(3) = [4,3,2]

  call basis(parameter_degrees)

end program main
