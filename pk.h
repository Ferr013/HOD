//Michele Trenti (2007)
#ifndef PK_HINCLUDED
#define PK_HINCLUDED

typedef struct pk_pass {
  int setnorm; // flag to indicate if normsig8 has been run
  float scale; // length scale to evaluate sigma2 at, in Mpc
  float anorm; // A in P(k) = A k^ns T(k)^2
  float ns; // primordial spectral index
  float sig8; // desired value of sigma_8
} PK;

float dsigtopdlnk(float lnk, void *ppk);
float mrsromb(float (*func)(float x, void *fdata), void *fdata,
            float a, float b);
float sigma2(float scale, PK *ppk);
int normsig8(float sigma8, float h, float ns, PK *ppk);
int pkinit(float omegam, float omegab, float h,
           float sigma8, float ns, PK *ppk);
float powerk(float k, PK *ppk);
float mdsigma2dm(float scale, PK *ppk);
float ddsigtopdmdlnk(float lnk, void *ppk);

#endif
