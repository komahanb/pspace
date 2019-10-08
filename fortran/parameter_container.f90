module parameter_container_class

  use abstract_parameter_class

  type param_ptr
     class(abstract_parameter), pointer :: p => null()
  end type param_ptr

  type :: parameter_container
     integer :: num_parameters = 0
     type(param_ptr), dimension(5) :: plist
   contains
     procedure :: add
     procedure :: basis
  end type parameter_container

contains

  subroutine add(this, param)

    class(parameter_container), intent(inout) :: this
    class(abstract_parameter) , target :: param

    if (this % num_parameters .gt. 5)  then
       print *, 'container full -- cannot add more parameters'
    end if

    this % num_parameters = this % num_parameters + 1
    this % plist (this % num_parameters) % p => param    

  end subroutine add

  real(8) function basis(this, z, d)

    class(parameter_container) , intent(in) :: this
    real(8)                    , intent(in) :: z(:)
    integer                    , intent(in) :: d(:)   
    integer :: ii

    nvars = size(z)

    basis = 1.0d0    
    do ii = 1, nvars
       basis = basis * this % plist(ii) % p % basis(z(ii), d(ii))       
    end do

  end function basis

end module parameter_container_class
