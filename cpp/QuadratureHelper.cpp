#include "QuadratureHelper.h"

QuadratureHelper::QuadratureHelper(){}
QuadratureHelper::~QuadratureHelper(){}

void QuadratureHelper::tensorProduct( const int nvars,
                                      const int *nqpts,
                                      const int **zp, const int **yp, const int **wp,
                                      int **zz, int **yy, int *ww ){  
  if (nvars == 1) {

    int ctr = 0;
    for (int ii = 0; ii < nqpts[0]; ii++){
      zz[0][ctr] = zp[0][ii];
      yy[0][ctr] = yp[0][ii];
      ww[ctr] = wp[0][ii];
      ctr++;
    }
    
  } else if (nvars == 2) {

    int ctr = 0;
    for (int ii = 0; ii < nqpts[0]; ii++){
      for (int jj = 0; jj < nqpts[1]; jj++){
        zz[0][ctr] = zp[0][ii];
        yy[0][ctr] = yp[0][ii];
        zz[1][ctr] = zp[1][jj];
        yy[1][ctr] = yp[1][jj];
        ww[ctr] = wp[0][ii]*wp[1][jj];
        ctr++;
      }
    }

  } else if (nvars == 3) {

    int ctr = 0;
    for (int ii = 0; ii < nqpts[0]; ii++){
      for (int jj = 0; jj < nqpts[1]; jj++){
        for (int kk = 0; kk < nqpts[2]; kk++){
          zz[0][ctr] = zp[0][ii];
          yy[0][ctr] = yp[0][ii];
          zz[1][ctr] = zp[1][jj];
          yy[1][ctr] = yp[1][jj];
          zz[2][ctr] = zp[2][kk];
          yy[2][ctr] = yp[2][kk];
          ww[ctr] = wp[0][ii]*wp[1][jj]*wp[2][kk];
          ctr++;
        }
      }
    }


  } else if (nvars == 4) {

    int ctr = 0;
    for (int ii = 0; ii < nqpts[0]; ii++){
      for (int jj = 0; jj < nqpts[1]; jj++){
        for (int kk = 0; kk < nqpts[2]; kk++){         
          for (int ll = 0; ll < nqpts[3]; ll++){
            zz[0][ctr] = zp[0][ii];
            yy[0][ctr] = yp[0][ii];
            zz[1][ctr] = zp[1][jj];
            yy[1][ctr] = yp[1][jj];
            zz[2][ctr] = zp[2][kk];
            yy[2][ctr] = yp[2][kk];
            zz[3][ctr] = zp[3][ll];
            yy[3][ctr] = yp[3][ll];
            ww[ctr] = wp[0][ii]*wp[1][jj]*wp[2][kk]*wp[3][ll];
            ctr++;
          }
        }
      }
    }
    
  } else if (nvars == 5) {

    int ctr = 0;
    for (int ii = 0; ii < nqpts[0]; ii++){
      for (int jj = 0; jj < nqpts[1]; jj++){
        for (int kk = 0; kk < nqpts[2]; kk++){         
          for (int ll = 0; ll < nqpts[3]; ll++){
            for (int mm = 0; mm < nqpts[4]; mm++){    
              zz[0][ctr] = zp[0][ii];
              yy[0][ctr] = yp[0][ii];
              zz[1][ctr] = zp[1][jj];
              yy[1][ctr] = yp[1][jj];
              zz[2][ctr] = zp[2][kk];
              yy[2][ctr] = yp[2][kk];
              zz[3][ctr] = zp[3][ll];
              yy[3][ctr] = yp[3][ll];
              zz[4][ctr] = zp[4][mm];
              yy[4][ctr] = yp[4][mm];              
              ww[ctr] = wp[0][ii]*wp[1][jj]*wp[2][kk]*wp[3][ll]*wp[4][mm];
              ctr++;
            }
          }
        }
      }
    }
    
  } // end if
  
}
