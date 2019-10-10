module parameter_container_class

  use abstract_parameter_class, only : abstract_parameter
  use basis_helper            , only : basis_degrees
  use quadrature_helper       , only : tensor_product
  
  implicit none

  type param_ptr
     class(abstract_parameter), pointer :: p => null()
  end type param_ptr

  type :: parameter_container

     integer :: num_parameters = 0

     type(param_ptr), dimension(5) :: plist
     integer        , dimension(5) :: param_maxdeg = 0
     ! integer        , dimension(5) :: param_nqpts = 0

     integer, allocatable :: dindex(:,:)
     real(8), allocatable :: zz(:,:), yy(:,:), ww(:)
     
     integer :: num_basis_terms = 0
     logical :: basis_initialized = .false.

     integer :: num_quadrature_points = 0
     logical :: quadrature_initialized = .false.
     
   contains

     procedure :: initialize_basis
     generic   :: basis => basis_term, basis_given_degrees
     
     procedure :: initialize_quadrature
     procedure :: get_quadrature
     
     procedure :: add
     procedure :: get_num_basis_terms
     procedure :: get_num_quadrature_points
     
     procedure, private :: basis_term, basis_given_degrees
     
  end type parameter_container

contains
  
  subroutine initialize_quadrature(this, pnqpts)

    class(parameter_container), intent(inout) :: this
    integer                   , intent(in) :: pnqpts(:)

    real(8) :: zp(this % num_parameters, maxval(pnqpts))
    real(8) :: yp(this % num_parameters, maxval(pnqpts))
    real(8) :: wp(this % num_parameters, maxval(pnqpts))
    integer :: p
    
    if (allocated(this % zz)) deallocate(this % zz)
    if (allocated(this % yy)) deallocate(this % yy)
    if (allocated(this % ww)) deallocate(this % ww) 
        
    do p = 1, this % num_parameters
       call this % plist (p) % p % quadrature(&
            & pnqpts(p), &
            & zp(p, 1 : pnqpts(p)), &
            & yp(p, 1 : pnqpts(p)), &
            & wp(p, 1 : pnqpts(p)) &
            & )
    end do

    call tensor_product(&
         & pnqpts, &
         & zp, yp, wp, &
         & this % zz, this % yy, this % ww)
    
    if (allocated(this % zz)) then 
       this % num_quadrature_points = size(this % zz, dim = 2)
    else
       print *, "quadrature points not allocated"
    end if
    
  end subroutine initialize_quadrature

  subroutine initialize_basis(this, pmax)

    class(parameter_container), intent(inout) :: this
    integer, intent(in) :: pmax(:)
    integer :: nterms, k

    !  Set the max degree for parameters
    this % param_maxdeg (1 : this % num_parameters) = pmax(:)

    ! generate and store a set of indices
    this % dindex = basis_degrees(this % param_maxdeg(1 : this % num_parameters))

    ! setup number of terms (assume tensor basis)
    this % num_basis_terms = size(this % dindex, dim=2)

    ! initialization flag true
    this % basis_initialized = .true.
    
  end subroutine initialize_basis

  impure subroutine add(this, param)

    class(parameter_container), intent(inout)      :: this
    class(abstract_parameter) , intent(in), target :: param

    if (this % num_parameters .gt. 5)  then
       print *, 'container full -- cannot add more parameters'
    end if

    this % num_parameters = this % num_parameters + 1
    this % plist (param % get_parameter_id()) % p => param        
    call this % plist(this % num_parameters) % p % print()

  end subroutine add

  pure integer function get_num_quadrature_points(this) result(nqpts)

    class(parameter_container) , intent(in) :: this

    nqpts = this % num_quadrature_points

  end function get_num_quadrature_points

  pure integer function get_num_basis_terms(this) result(nterms)
    
    class(parameter_container) , intent(in) :: this
    
    nterms = this % num_basis_terms
    
  end function get_num_basis_terms
  
  real(8) function basis_term(this, k, z) result(psi)

    class(parameter_container) , intent(in) :: this
    real(8)                    , intent(in) :: z(:)
    integer                    , intent(in) :: k
    integer :: ii, nvars
    
    associate(d=>this % dindex(:,k))
      psi = 1.0d0    
      do ii = 1, this % num_parameters
         psi = psi * this % plist(ii) % p % basis(z(ii), d(ii))
      end do
    end associate

  end function basis_term

  real(8) function basis_given_degrees(this, d, z) result(psi)

    class(parameter_container) , intent(in) :: this
    real(8)                    , intent(in) :: z(:)
    integer                    , intent(in) :: d(:)   
    integer :: ii

    psi = 1.0d0    
    do ii = 1, this % num_parameters
       psi = psi * this % plist(ii) % p % basis(z(ii), d(ii))       
    end do

  end function basis_given_degrees
  
  subroutine get_quadrature(this, q, z, y, w)

    class(parameter_container) , intent(in)    :: this
    integer                    , intent(in)    :: q
    real(8)                    , intent(inout) :: z(:), y(:)
    real(8)                    , intent(inout) :: w

    z = this % zz(:,q)
    y = this % yy(:,q)
    w = this % ww(q)

  end subroutine get_quadrature

end module parameter_container_class
