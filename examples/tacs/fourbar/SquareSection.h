#include "TACSTimoshenkoConstitutive.h"

class SquareSection : public TACSTimoshenkoConstitutive {
public:
  static const double kcorr = 5.0/6.0;
  
  SquareSection( TacsScalar _density, TacsScalar _E, TacsScalar _G,
                 TacsScalar _w, int _wNum,
                 const TacsScalar axis[] ):
    TACSTimoshenkoConstitutive(NULL, NULL, axis){
    density = _density;
    E = _E;
    G = _G;
    w = _w;
    wNum = _wNum;
    if (wNum < 0){
      wNum = 0;
    }

    computeProperties();
  }

  void computeProperties(){
    // Set the properties based on the width/thickness variables
    TacsScalar A = w*w;
    TacsScalar Iy = w*w*w*w/12.0;
    TacsScalar Iz = Iy;
    TacsScalar J = Iy + Iz;
    TacsScalar Iyz = 0.0;

    // Set the entries of the stiffness matrix
    memset(C, 0, 36*sizeof(TacsScalar));
    C[0] = E*A;
    C[7] = G*J;
    C[14] = E*Iy;
    C[21] = E*Iz;
    C[28] = kcorr*G*A;
    C[35] = kcorr*G*A;

    // Set the entries of the density matrix
    rho[0] = density*A;
    rho[1] = density*Iy;
    rho[2] = density*Iz;
    rho[3] = density*Iyz;
  }

  int getDesignVarNums( int elemIndex, int dvLen, int dvNums[] ){
    if (dvNums){
      dvNums[0] = wNum;
    }
    return 1;
  }
  int setDesignVars( int elemIndex, int dvLen, const TacsScalar dvs[] ){
    w = dvs[0];
    computeProperties();
    return 1;
  }
  int getDesignVars( int elemIndex, int dvLen, TacsScalar dvs[] ){
    dvs[0] = w;
    return 1;
  }
  int getDesignVarRange( int elemIndex, int dvLen,
                         TacsScalar lb[], TacsScalar ub[] ){
    lb[0] = 0.0;
    ub[0] = 10.0;
    return 1;
  }
  void addStressDVSens( int elemIndex, const double pt[], const TacsScalar X[],
                        const TacsScalar e[], TacsScalar scale,
                        const TacsScalar psi[], int dvLen, TacsScalar dfdx[] ){
    dfdx[0] += scale*(2.0*w*(E*e[0]*psi[0] + kcorr*G*(e[4]*psi[4] + e[5]*psi[5])) +
                      (w*w*w/3.0)*(2.0*G*e[1]*psi[1] + E*(e[2]*psi[2] + e[3]*psi[3])));
  }
  void addMassMomentsDVSens( int elemIndex, const double pt[],
                             TacsScalar scale, const TacsScalar psi[],
                             int dvLen, TacsScalar dfdx[] ){
    dfdx[0] += scale*density*(2.0*w*psi[0] + ((w*w*w)/3.0)*(psi[1] + psi[2]));
  }

  TacsScalar evalDensity( int elemIndex, const double pt[],
                          const TacsScalar X[] ){
    return density*w*w;
  }
  void addDensityDVSens( int elemIndex, const double pt[],
                         const TacsScalar X[], const TacsScalar scale,
                         int dvLen, TacsScalar dfdx[] ){
    dfdx[0] += 2.0*scale*density*w;
  }

  TacsScalar evalFailure( int elemIndex, const double pt[],
                          const TacsScalar X[], const TacsScalar e[] ){
    return E*w*w*fabs(e[0])/10e3;
  }
  TacsScalar evalFailureStrainSens( int elemIndex, const double pt[],
                                    const TacsScalar X[], const TacsScalar e[],
                                    TacsScalar sens[] ){
    memset(sens, 0, 6*sizeof(TacsScalar));
    if (TacsRealPart(e[0]) >= 0.0){
      sens[0] = E*w*w/10e3;
    }
    else {
      sens[0] = -E*w*w/10e3;
    }
    return E*w*w*fabs(e[0])/10e3;
  }
  void addFailureDVSens( int elemIndex, const double pt[],
                         const TacsScalar X[], const TacsScalar e[],
                         TacsScalar scale, int dvLen, TacsScalar dfdx[] ){
    dfdx[0] += 2.0*scale*E*w*fabs(e[0])/10e3;
  }

  TacsScalar density, E, G;
  TacsScalar w;
  int wNum;
};
