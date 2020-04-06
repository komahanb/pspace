#include <stdio.h>
#include"QuadratureHelper.h"

#include"NormalParameter.h"
#include"UniformParameter.h"
#include"ExponentialParameter.h"

/**
   Constructor for quadrature helper

   @param quadrature_type used for variants of quadrature type
 */
QuadratureHelper::QuadratureHelper(int _quadrature_type){
  this->quadrature_type = _quadrature_type;
}

/**
   Destructor for quadrature helper
 */
QuadratureHelper::~QuadratureHelper(){}

/**
   Function that performs tensor product of univariate rules

   @param nvars number of variables
   @param nqpts number of quadrature points for each variable
   @param zp quadrature point in standardized probabilistic domain
   @param yp quadrature point in general probabilistic domain
   @param wp weights for each quadrature point
   @param zz multivariate quadrature points in standard domain
   @param yy multivariate quadrature points in general domain
   @param ww multivariate quadrature weights
 */
void QuadratureHelper::tensorProduct( const int nvars,
                                      const int *nqpts,
                                      scalar **zp, scalar **yp, scalar **wp,
                                      scalar **zz, scalar **yy, scalar *ww ){


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

/**
   Test of quadrature construction
 */
void test_quadraure_multi(int argc, char *argv[] ){

  QuadratureHelper *qh = new QuadratureHelper();

  const int nvars = 3;

  int nqpts[nvars];
  nqpts[0] = 4;
  nqpts[1] = 2;
  nqpts[2] = 3;

  scalar **zp = new scalar*[nvars];
  for (int i = 0; i < nvars; i++){
    zp[i] = new scalar[nqpts[0]];
  }

  scalar **yp = new scalar*[nvars];
  for (int i = 0; i < nvars; i++){
    yp[i] = new scalar[nqpts[1]];
  }

  scalar **wp = new scalar*[nvars];
  for (int i = 0; i < nvars; i++){
    wp[i] = new scalar[nqpts[2]];
  }

  NormalParameter *p1 = new NormalParameter(0, 0.0, 1.0);
  UniformParameter *p2 = new UniformParameter(1, -1.0, 1.0);
  ExponentialParameter *p3 = new ExponentialParameter(2, 1.0, 0.5);

  p1->quadrature(nqpts[0], zp[0], yp[0], wp[0]);
  p2->quadrature(nqpts[1], zp[1], yp[1], wp[1]);
  p3->quadrature(nqpts[2], zp[2], yp[2], wp[2]);

  scalar wsum = 0.0;
  for(int k = 0; k < nqpts[0]; k++){
    wsum += wp[0][k];
    printf("%f %f %f %f \n", RealPart(zp[0][k]), RealPart(yp[0][k]), RealPart(wp[0][k]), RealPart(wsum));
  }

  printf("\n");
  wsum = 0.0;
  for(int k = 0; k < nqpts[1]; k++){
    wsum += wp[1][k];
    printf("%f %f %f %f \n", RealPart(zp[1][k]), RealPart(yp[1][k]), RealPart(wp[1][k]), RealPart(wsum));
  }

  printf("\n");
  wsum = 0.0;
  for(int k = 0; k < nqpts[2]; k++){
    wsum += wp[2][k];
    printf("%f %f %f %f\n", RealPart(zp[2][k]), RealPart(yp[2][k]), RealPart(wp[2][k]), RealPart(wsum));
  }

  scalar **zz;
  scalar **yy;
  scalar *ww;

  // Allocate space for return variables
  int totquadpts = 1;
  for (int i = 0; i < nvars; i++){
    totquadpts *= nqpts[i];
  }

  zz = new scalar*[nvars];
  yy = new scalar*[nvars];
  for (int i = 0; i < nvars; i++){
    zz[i] = new scalar[totquadpts];
    yy[i] = new scalar[totquadpts];
  }
  ww = new scalar[totquadpts];

  // Find tensor product of 1d rules
  qh->tensorProduct(nvars, nqpts,
                    zp, yp, wp,
                    zz, yy, ww);

  wsum = 0.0;
  for(int k = 0; k < totquadpts; k++){
    printf("%d ", k);
    for(int j = 0; j < nvars; j++){
      printf(" %f ", RealPart(zz[j][k]));
    }
    wsum += ww[k];
    printf(" %f %f \n", RealPart(ww[k]), RealPart(wsum));
  }

}
