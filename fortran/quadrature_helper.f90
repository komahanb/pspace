module quadrature_helper

  implicit none

contains
  
  subroutine tensor_product( &
       & nqpts, &
       & zp, yp, wp, &
       & zz, yy, ww)

    integer, intent(in) :: nqpts(:)
    real(8), intent(in) :: zp(:,:), yp(:,:), wp(:,:)
    real(8), allocatable, intent(inout) :: zz(:,:), yy(:,:), ww(:)

    integer :: ii, jj, kk, ll, mm, ctr, nvars

    nvars = size(zp, dim = 1)    
    allocate(zz(nvars,product(nqpts)))
    allocate(yy(nvars,product(nqpts)))
    allocate(ww(product(nqpts)))

    if (nvars .eq. 1) then

       ctr = 1
       do ii = 1, nqpts(1)
          zz(:,ctr) = [zp(1,ii)]
          yy(:,ctr) = [yp(1,ii)]
          ww(ctr) = wp(1,ii)
          ctr = ctr + 1
       end do

    else if (nvars .eq. 2) then

       ctr = 1
       do ii = 1, nqpts(1)
          do jj = 1, nqpts(2)
             zz(:,ctr) = [zp(1,ii), zp(2,jj)]
             yy(:,ctr) = [yp(1,ii), yp(2,jj)]
             ww(ctr) = wp(1,ii)*wp(2,jj)
             ctr = ctr + 1
          end do
       end do

    else if (nvars .eq. 3) then

       ctr = 1
       do ii = 1, nqpts(1)
          do jj = 1, nqpts(2)
             do kk = 1, nqpts(3)
                zz(:,ctr) = [zp(1,ii), zp(2,jj), zp(3,kk)]
                yy(:,ctr) = [yp(1,ii), yp(2,jj), yp(3,kk)]
                ww(ctr) = wp(1,ii)*wp(2,jj)*wp(3,kk)
                ctr = ctr + 1
             end do
          end do
       end do

    else if (nvars .eq. 4) then

       ctr = 1
       do ii = 1, nqpts(1)
          do jj = 1, nqpts(2)
             do kk = 1, nqpts(3)
                do ll = 1, nqpts(4)
                   zz(:,ctr) = [zp(1,ii), zp(2,jj), zp(3,kk), zp(4,ll)]
                   yy(:,ctr) = [yp(1,ii), yp(2,jj), yp(3,kk), yp(4,ll)]
                   ww(ctr) = wp(1,ii)*wp(2,jj)*wp(3,kk)*wp(4,ll)
                   ctr = ctr + 1
                end do
             end do
          end do
       end do

    else if (nvars .eq. 5) then

       ctr = 1
       do ii = 1, nqpts(1)
          do jj = 1, nqpts(2)
             do kk = 1, nqpts(3)
                do ll = 1, nqpts(4)
                   do mm = 1, nqpts(5)
                      zz(:,ctr) = [zp(1,ii), zp(2,jj), zp(3,kk), zp(4,ll), zp(5,mm)]
                      yy(:,ctr) = [yp(1,ii), yp(2,jj), yp(3,kk), yp(4,ll), yp(5,mm)]
                      ww(ctr) = wp(1,ii)*wp(2,jj)*wp(3,kk)*wp(4,ll)*wp(5,mm)
                      ctr = ctr + 1
                   end do
                end do
             end do
          end do
       end do

    end if

  end subroutine tensor_product
  
end module quadrature_helper
