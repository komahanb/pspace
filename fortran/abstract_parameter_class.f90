module abstract_parameter_class

  !===================================================================!
  ! Abstract parameter type
  !===================================================================!
  
  type, abstract :: abstract_parameter

     integer :: parameter_id

   contains

     procedure :: get_parameter_id
     procedure :: set_parameter_id

     ! Deferred procedure
     procedure(quadrature_interface), deferred :: quadrature
     procedure(basis_interface)     , deferred :: basis     

  end type abstract_parameter

  interface

     subroutine quadrature_interface(this, npoints, z, y, w)
       import abstract_parameter
       class(abstract_parameter) , intent(in)    :: this
       integer                   , intent(in)    :: npoints
       real(8)                   , intent(inout) :: z(:), y(:)
       real(8)                   , intent(inout) :: w(:)
     end subroutine quadrature_interface

     real(8) function basis_interface(this, z, d)
       import abstract_parameter
       class(abstract_parameter) , intent(in) :: this
       real(8)                   , intent(in) :: z
       integer                   , intent(in) :: d
     end function basis_interface

  end interface

contains

  pure subroutine set_parameter_id(this, parameter_id)

    class(abstract_parameter), intent(inout) :: this
    integer, intent(in) :: parameter_id

    this % parameter_id = parameter_id

  end subroutine set_parameter_id

  type(integer) function get_parameter_id(this)

    class(abstract_parameter), intent(in) :: this

    get_parameter_id =  this % parameter_id

  end function get_parameter_id

end module abstract_parameter_class
