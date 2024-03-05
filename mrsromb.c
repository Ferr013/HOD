/* mrsromb
 *
 * Closed Romberg integrator, based on Numerical Recipes qromb
 *
 * First I excised qromb from the NR library along with required
 * support routines, so that it can be compiled stand-alone.
 *
 * The only change I made to the actual code is the passing through of
 * the void pointer fdata.  This is provided in case the user wants to
 * evaluate a function that depends on more than one variable.  This
 * is not the best-performing way to implement this, but it is
 * reasonably flexible.  An example is shown in exmrsromb.c.
 *
 * mrsromb is generally thread safe itself.  However, it is the user's
 * responsibility to ensure the thread safety of the integrand
 * function, specifically through its potential reference to fdata.
 *
 */
//Michele Trenti (2007)

/* prototypes */
float mrsromb(float (*func)(float x, void *fdata), void *fdata,
            float a, float b);
float trapzd(float (*func)(float x, void *fdata), void *fdata,
             float a, float b, int n);
void nrerror(char error_text[]);
float *vector(long nl, long nh);
void free_vector(float *v, long nl, long nh);
void polint(float xa[], float ya[], int n, float x, float *y, float *dy);

/* qromb from NumRec */

#include <math.h>
#define EPS 1.0e-5 // default was 1.0e-6
#define JMAX 15 // default was 20
#define JMAXP (JMAX+1)
#define K 5

float mrsromb(float (*func)(float x, void *fdata), void *fdata,
            float a, float b)
{
  float ss,dss;
  float s[JMAXP],h[JMAXP+1];
  int j;

  h[1]=1.0;
  for (j=1;j<=JMAX;j++) {
    s[j]=trapzd(func,fdata,a,b,j);
    if (j >= K) {
      polint(&h[j-K],&s[j-K],K,0.0,&ss,&dss);
      if (fabs(dss) <= EPS*fabs(ss)) return ss;
    }
    h[j+1]=0.25*h[j];
  }
  nrerror("Too many steps in routine mrsromb");
  return 0.0;
}
#undef EPS
#undef JMAX
#undef JMAXP
#undef K

/* trapzd from NumRec */

#define FUNC(x) ((*func)(x,fdata))

float trapzd(float (*func)(float x, void *fdata), void *fdata,
             float a, float b, int n)
{
	float x,tnm,sum,del;
	static float s;
	int it,j;

	if (n == 1) {
		return (s=0.5*(b-a)*(FUNC(a)+FUNC(b)));
	} else {
		for (it=1,j=1;j<n-1;j++) it <<= 1;
		tnm=it;
		del=(b-a)/tnm;
		x=a+0.5*del;
		for (sum=0.0,j=1;j<=it;j++,x+=del) sum += FUNC(x);
		s=0.5*(s+(b-a)*sum/tnm);
		return s;
	}
}
#undef FUNC

/* polint from NumRec (plus required elements of nrutils.c/h) */

#include <math.h>
#define NRANSI
#include <stdio.h>
#include <stdlib.h>
#define NR_END 1
#define FREE_ARG char*

void nrerror(char error_text[])
     /* Numerical Recipes standard error handler */
{
  fprintf(stderr,"Numerical Recipes run-time error...");
  fprintf(stderr,"%s\n",error_text);
  //  fprintf(stderr,"...now exiting to system...\n");
  //  exit(1);
}

float *vector(long nl, long nh)
     /* allocate a float vector with subscript range v[nl..nh] */
{
  float *v;

  v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
  if (!v) nrerror("allocation failure in vector()");
  return v-nl+NR_END;
}

void free_vector(float *v, long nl, long nh)
     /* free a float vector allocated with vector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}

void polint(float xa[], float ya[], int n, float x, float *y, float *dy)
{
  int i,m,ns=1;
  float den,dif,dift,ho,hp,w;
  float *c,*d;

  dif=fabs(x-xa[1]);
  c=vector(1,n);
  d=vector(1,n);
  for (i=1;i<=n;i++) {
    if ( (dift=fabs(x-xa[i])) < dif) {
      ns=i;
      dif=dift;
    }
    c[i]=ya[i];
    d[i]=ya[i];
  }
  *y=ya[ns--];
  for (m=1;m<n;m++) {
    for (i=1;i<=n-m;i++) {
      ho=xa[i]-x;
      hp=xa[i+m]-x;
      w=c[i+1]-d[i];
      if ( (den=ho-hp) == 0.0) nrerror("Error in routine polint");
      den=w/den;
      d[i]=hp*den;
      c[i]=ho*den;
    }
    *y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
  }
  free_vector(d,1,n);
  free_vector(c,1,n);
}
#undef NRANSI
