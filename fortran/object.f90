module object_class

  type :: object
     integer :: id
  end type object

  interface object
     module procedure create
  end interface

contains
  
  function create(id) result(this)
    integer, intent(in) :: id
    type(object) :: this
    this % id = id
  end function create

end module object_class

program test_object
  use object_class
  type(object) :: o
   o = object(10)
end program test_object
