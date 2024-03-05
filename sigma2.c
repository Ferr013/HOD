//Michele Trenti (2007)

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "pk.h"

#define MIN(a,b) ((a)<(b) ? (a) : (b))
#define SIG2EPS 1e-5
#define UPPERZERO 30

// initialize both transfer function and P(k)
int pkinit(float omegam, float omegab, float h,
           float sigma8, float ns, PK *ppk)
{
  normsig8(sigma8,h,ns,ppk);
  return(EXIT_SUCCESS);
}

// normalize power spectrum by sigma_8
int normsig8(float sigma8, float h, float ns, PK *ppk)
{
  float scale8;
  float sig2;

  if (ppk->setnorm) {
    fprintf(stderr,"P(k) parameters already set (or bad PK initialization)\n");
    exit(EXIT_FAILURE);
  }

  ppk->setnorm = 1;

  scale8 = 8/h; // 8 h^-1 Mpc

  ppk->ns = ns;

  ppk->anorm = 1e7; // dummy value, but keeps the sigma_8 integral around unity

  sig2 = sigma2(scale8, ppk);

  ppk->anorm *= sigma8*sigma8/sig2;

  ppk->sig8 = sigma2(scale8, ppk);

  return(EXIT_SUCCESS);
}

// P(k)
float powerk(float k, PK *ppk)
{
  float anorm,ns,tf;

  if (!ppk->setnorm) {
    fprintf(stderr,"Must set P(k) parameters before calling pk\n");
    exit(EXIT_FAILURE);
  }

  anorm = ppk->anorm;
  ns = ppk->ns;
  //tf = mrstf(k,&(ppk->eh));

  //     P(k)
  //     A     * k^ns      * T(k)^2
  return anorm * pow(k,ns) * tf*tf;
}

float sigma2(float scale, PK *ppk)
{
  #include "bessel.h"
  float lnk1,lnk2;
  float lnkeq,lnkwlow;
  float lnklow1=0,lnklow2,lnkhigh1,lnkhigh2=0;
  float sigma2,totsig;
  unsigned char flag_trylow,flag_tryhigh;
  unsigned int upperzero;
  if (!ppk->setnorm) {
    fprintf(stderr,"Must set P(k) parameters before calling sigma2\n");
    exit(EXIT_FAILURE);
  }
  ppk->scale = scale;
  //lnkeq = log((ppk->eh).k_equality) - 3;
  lnkwlow = log(1/ppk->scale) - 4;
  lnk1 = MIN(lnkeq,lnkwlow);
  upperzero = UPPERZERO;
  lnk2 = log(besselzero[upperzero]/ppk->scale);

  sigma2 = mrsromb(dsigtopdlnk,ppk,lnk1,lnk2);

  totsig = sigma2;
  flag_trylow = 1;
  flag_tryhigh = 1;
  lnklow2 = lnk1; // set upper limit of lower expansion to initial lower limit
  lnkhigh1 = lnk2; // set lower limits of upper expansion to initial upper lim

  while(flag_trylow || flag_tryhigh) {
    if (flag_trylow) {
      lnklow1 = lnklow2 - 1; // use \Delta ln k = 1
      sigma2 = mrsromb(dsigtopdlnk,ppk,lnklow1,lnklow2);
      totsig += sigma2;
      lnklow2 = lnklow1; // reset upper limit for next time
      if (sigma2<SIG2EPS*totsig) {
        flag_trylow = 0;
      }
    }
    if (flag_tryhigh) {
      upperzero += 1;
      if (upperzero>=NBESSEL) {
        fprintf(stderr,"Need more bessel zeros\n");
        exit(EXIT_FAILURE);
      }
      lnkhigh2 = log(besselzero[upperzero]/ppk->scale); // int to next zero
      sigma2 = mrsromb(dsigtopdlnk,ppk,lnkhigh1,lnkhigh2);
      totsig += sigma2;
      lnkhigh1 = lnkhigh2; // reset upper limit for next time
      if (sigma2<SIG2EPS*totsig) {
        flag_tryhigh = 0;
      }
    }
  }
  return totsig;
}

/* Integrand of spherical top-hat density variance integral, expressed
   per ln k interval, as a function of ln k.

   The polynomial approximation to the window function at small k is
   required for accuracy (presumably it is also much faster).  I have
   verified that it returns identical results to the proper Bessel
   function value over the range I use it, at the precision of float
   (i.e., if you make it double then you need to recheck this).
*/
float dsigtopdlnk(float lnk, void *ppk)
{
#define MRS_1_2PI2 0.05066059182116889f /* 1/(2 pi^2) */
  float k;
  float scale;
  float anorm;
  float ns;
  float x,x2,x4,x6,x8,x10;
  float tf;
  float w,w2;
  k = exp((double) lnk);
  scale = ((PK *)ppk)->scale;
  x = scale * k;
  //tf = mrstf(k,&(((PK *)ppk)->eh));
  if (x<1) { // use polynomial expansion
    /*    w2 = 1 - x*x/5 + 3*pow(x,4)/175 - 4*pow(x,6)/4725
          + 2*pow(x,8)/72765 - pow(x,10)/1576575;*/
    x2 = x*x; // this replacement actually speeds this subroutine up a lot
    x4 = x2*x2;
    x6 = x4*x2;
    x8 = x4*x4;
    x10 = x6*x4;
    //    w2 = 1 - x2/5 + 3*x4/175 - 4*x6/4725 + 2*x8/72765 - x10/1576575;
    w2 = 1 - x2*0.2 + 3*x4/175 - 4*x6/4725 + 2*x8/72765 - x10/1576575;
  } else { // use full spherical bessel function
    w = 3.f*(x*cos((double) x) - sin((double) x))/(x*x*x);
    w2 = w*w;
  }
  anorm = ((PK *)ppk)->anorm;
  ns = ((PK *)ppk)->ns;
  //     1/(2 pi^2) * P(k)                       * W(k)^2 * k^2 * dk/dlnk
  //     1/(2 pi^2) * A     * k^ns      * T(k)^2 * W(k)^2 * k^2 * k
  return MRS_1_2PI2 * anorm * pow((double)k,ns) * tf * tf * w2 * k * k * k;

}

/* Integrand of derivative of spherical top-hat density (w.r.t. mass,
   but see below) variance integral, expressed per ln k interval, as a
   function of ln k.

   The derivative introduces a factor of mass into the integrand.
   However, since this doesn't depend on k it can be moved outside of
   the integral, which I have done.  All the other factors are kept in
   here.

   The polynomial approximation to the window function at small k is
   required for accuracy, and is good to better than 1e-9 accuracy.

*/
float ddsigtopdmdlnk(float lnk, void *ppk)
{
#define MRS_1_2PI2 0.05066059182116889f /* 1/(2 pi^2) */
  float k;
  float scale;
  float anorm;
  float ns;
  float x,x2,x4,x6,x8,x10;
  float tf;
  float sinx,cosx,wfact;

  k = exp((double) lnk);
  scale = ((PK *)ppk)->scale;
  x = scale * k;
  // tf = mrstf(k,&(((PK *)ppk)->eh));

  // d/dM = dx/dR dR/dM d/dx = k R/3M d/dx = x/3M d/dx
  // wfact = x/3 * d/dx W^2(x) = 2x/3 W(x) dW/dx
  if (x<0.25) { // use polynomial expansion
    x2 = x*x;
    x4 = x2*x2;
    x6 = x4*x2;
    x8 = x4*x4;
    x10 = x6*x4;
    wfact = -2.f*x2/15.f + 4.f*x4/175.f - 8.*x6/4725.f + 16*x8/218295.f - 2.f*x10/945945.f;
  } else { // use full trig functions
    cosx = cos((double) x);
    sinx = sin((double) x);
    wfact = -6.f * (x*cosx - sinx) * (3.f*x*cosx + (x*x - 3.f)*sinx) / (x*x*x*x*x*x);
  }
  anorm = ((PK *)ppk)->anorm;
  ns = ((PK *)ppk)->ns;
  //     1/(2 pi^2) * P(k)                  * x/3*2*W(x)*dW/dx * k^2 * dk/dlnk
  return MRS_1_2PI2 * anorm*pow((double) k,ns) * tf * tf * wfact * k * k * k;

}
