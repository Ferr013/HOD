import os.path
import numpy as np
from scipy import integrate
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
import mpmath

from tqdm.notebook import tqdm

from hmf import MassFunction
from halomod.bias import Tinker10
from astropy.cosmology import FlatLambdaCDM
cosmo  = FlatLambdaCDM(H0=67.74, Om0=0.3089, Tcmb0=2.725)
OmegaM = cosmo.Om(0)
OmegaL = cosmo.Ode(0)
OmegaK = cosmo.Ok(0)
OmegaB = 0.049
OmegaC = OmegaM-OmegaB
H0 = cosmo.H(0).value
h  = H0/100
s8 = 0.9
c_light  = 299792.458 #speed of light km/s
G  = 4.3009e-9  #Mpc/Msolar*(km/s)^2
Theta=2.728/2.7 #Eisenstein_Hu_98
zeq=2.5*10**4*OmegaM*h**2*Theta**(-4) #Eisenstein_Hu_98

###################################################################################################
### LINEAR POWER SPECTRUM #########################################################################
def rho_m(z = 0):
    return cosmo.Om(z)*cosmo.critical_density(z).value*1.477e40# M_sun/Mpc^3

def D_growth_factor(z):
    #Eisenstein & Hu (1999)
    integrand= lambda x: (1+x)/(OmegaM*(1+x)**3+(1-OmegaM-OmegaL)*(1+x)**2+OmegaL)**1.5
    Growth=integrate.quad(integrand,z,zeq)
    norm=integrate.quad(integrand,0,zeq)
    N=norm[0]
    GrowthFactor=(OmegaM*(1+z)**3+(1-OmegaM-OmegaL)*(1+z)**2+OmegaL)**0.5*Growth[0]/N
    return GrowthFactor

def T(k): #EH99
    s=44.5*np.log(9.83/(OmegaM*h**2))/(1+10*(OmegaB*h**2)**0.75)**0.5
    fc=OmegaC/OmegaM
    fb=OmegaB/OmegaM
    fcb=fc+fb
    alpha=fc/fcb
    Gamma=OmegaM*h**2*(alpha**0.5+(1-alpha**0.5)/(1+(0.43*k*s)**4))
    q=k*Theta**2/Gamma
    Beta=1./(1-0.949*fb)
    L=np.log(np.exp(1)+1.84*Beta*alpha**0.5*q)
    C=14.4+(325./(1+60.5*q**1.11))
    Tsup=L/(L+C*q**2)
    Tmaster=Tsup
    return Tmaster

def norm_power_spectrum():
    n = 0.96
    N = n-1
    deltah = 1.94*10**(-5)*OmegaM**(-0.785-0.05*np.log(OmegaM))*np.exp(-0.95*N-0.169*N**2)
    intgr_n = lambda x: (2*(np.pi*deltah*T(x))**2*(c_light*x/H0)**(3+n)/(x**3))/\
                        (2*np.pi**2)*np.power(x,2)*W_F_transf(x, 8/h)**2
    norm = integrate.quad(intgr_n, 0, 1000)[0]
    return (s8**2)/norm

def power_spectrum(k, z, D_ratio, _PS_NORM_):
    n = 0.96
    N = n-1
    deltah = 1.94*10**(-5)*OmegaM**(-0.785-0.05*np.log(OmegaM))*np.exp(-0.95*N-0.169*N**2)
    return 2*(np.pi*deltah*T(k))**2*(c_light*k/H0)**(3+n)/(k**3)*D_ratio*_PS_NORM_

###################################################################################################
### HALO MASS FUNCTION ############################################################################

def W_F_transf(k, r):
    kr = np.outer(k, r)
    return (3 / kr**3) * (np.sin(kr) - kr * np.cos(kr))

def dW_dlnR(k, r):
    kr = np.outer(k, r)
    return (9 * kr * np.cos(kr) + 3 * (kr**2 - 3) * np.sin(kr)) / kr**3

def _sigma(r, z, D_ratio, _PS_NORM_):
    #intgr = lambda x: power_spectrum(x, z, D_ratio, _PS_NORM_)/
    #                  \(2*np.pi**2)*np.power(x,2)*W_F_transf(x, r)**2
    #sigma_sq = integrate.quad(intgr, 0, 1000)[0]
    k_array = np.logspace(-4, 4, 1000)
    dlogk = np.log(k_array[1]/k_array[0])
    intgrnd = np.zeros(0)
    for x in k_array:
        _WfT = np.power(x,2)*W_F_transf(x, r)**2/(2*np.pi**2)
        intgrnd = np.append(intgrnd, power_spectrum(x, z, D_ratio, _PS_NORM_) * _WfT)
    sigma_sq = np.sum(k_array*intgrnd) * dlogk
    return np.sqrt(sigma_sq)

def _dlnsdlnm(r, sigma, z, D_ratio, _PS_NORM_):
    #intgr = lambda x: power_spectrum(x, z, D_ratio, _PS_NORM_)
    #                  \*np.power(x,2)*W_F_transf(x, r)*dW_dlnR(x, r)
    #I_R = integrate.quad(intgr, 0, 1000, limit=2000)[0]
    k_array = np.logspace(-4, 4, 1000)
    dlogk = np.log(k_array[1]/k_array[0])
    intgrnd = np.zeros(0)
    for x in k_array:
        _Wterm = np.power(x,2)*W_F_transf(x, r)*dW_dlnR(x, r)
        intgrnd = np.append(intgrnd, power_spectrum(x, z, D_ratio, _PS_NORM_) * _Wterm)
    I_R = np.sum(k_array*intgrnd) * dlogk
    return 1 / (6*np.pi**2) * np.abs(I_R) / sigma**2

def f_TINKER(sigma, z): #for \Delta = 200
	A0, a0, b0, c0 = 0.186, 1.47, 2.57, 1.19
	alpha = 0.0106756286522959 #10**(-(0.75 / np.log10(200 / 75.0))**1.2)
	A = A0 * (1.0 + z)**-0.140
	a = a0 * (1.0 + z)**-0.060
	b = b0 * (1.0 + z)**-alpha
	return A * (np.power(sigma / b,-a) + 1.0) * np.exp(-c0 / sigma**2)

def nuc(sigma, z): #number of standard deviation
    deltacrit = 1.69 / D_growth_factor(z)
    return deltacrit / sigma

def HMF(M_h, z):
    _PS_NORM_ = norm_power_spectrum()
    D_ratio   = (D_growth_factor(z)/D_growth_factor(0))**2 if z != 0 else 1
    r = np.power(3.0 * M_h / (4.0 * np.pi * rho_m(z)), 1/3)
    sigma = _sigma(r, z, D_ratio, _PS_NORM_)
    f_sigma = f_TINKER(sigma, z)
    return f_sigma * rho_m(z) * _dlnsdlnm(r, sigma, z, D_ratio, _PS_NORM_) / M_h**2
    #return np.sqrt(2./np.pi) * rho_m() / M_h**2 * _dlnsdlnm(r, sigma, z, D_ratio, _PS_NORM_)\
    #       * nuc(sigma, z) * np.exp(-(nuc(sigma, z))/2) #dn/dlnm

##################################################################################################
### HALO OCCUPATION DISTRIBUTION #################################################################

def N_cen(M_h, M_min, sigma_logM):
    return 0.5*(1+special.erf((np.log10(M_h)-np.log10(M_min))/(np.sqrt(2)*sigma_logM)))

def N_sat(M_h, M_sat, alpha, M_min, sigma_logM):
    M_cut = np.power(M_min, -0.5)
    return  N_cen(M_h, M_min, sigma_logM)*np.power((M_h-M_cut)/M_sat,alpha)

def N_tot(M_h, M_sat, alpha, M_min, sigma_logM):
    return N_cen(M_h, M_min, sigma_logM) + N_sat(M_h, M_sat, alpha, M_min, sigma_logM)
    #R = np.zeros(len(M_h))
    #R[M_h>M_min] = np.power(M_h[M_h>M_min]/M_sat, alpha)
    #return R

def n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array):
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    return np.trapz(HMF_array*NTOT, M_h_array)
    #dlogk = np.log(M_h_array[1]/M_h_array[0])
    #return np.sum(M_h_array*HMF_array*NTOT) * dlogk

def get_c_from_M_h(M_h, z):
    if(0):
        #Eq.4 https://iopscience.iop.org/article/10.1086/367955/pdf
        c_norm = 8
        return (c_norm) / (1+z) * np.power(M_h / (1.4e14), -0.13)
    else:
        #DUFFY ET AL: Eq.4 https://arxiv.org/pdf/0804.2486.pdf
        M_pivot = 2e12/h #M_sun
        A, B, C = 6.71, -0.091, -0.44 #Relaxed
        #A, B, C = 5.71, -0.084, -0.47 #Full
        return A * np.power(M_h / M_pivot, B) * (1+z) ** C

def u_FT(k, M_h, z, crit_dens_rescaled):
    r_v = np.power(M_h/crit_dens_rescaled, 1/3) #rho = M_sun/Mpc^3
    c   = get_c_from_M_h(M_h, z)
    f_c = np.log(1+c)-c/(1+c)
    r_s = r_v/c
    si, ci = special.sici(k*r_s)
    si_c, ci_c = special.sici(k*r_s*(1+c))
    return (np.sin(k*r_s)*(si_c-si)+np.cos(k*r_s)*(ci_c-ci)-(np.sin(c*k*r_s)/((1+c)*k*r_s)))/f_c

def PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z,
             crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT):
    U_FT = u_FT(k, M_h_array, z, crit_dens_rescaled)
    intg = np.trapz(HMF_array * NCEN * NSAT * U_FT, M_h_array)
    return intg * 2 / np.power(N_G, 2)

def PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z,
             crit_dens_rescaled, M_h_array, HMF_array, N_G, NSAT):
    U_FT = u_FT(k, M_h_array, z, crit_dens_rescaled)
    intg = np.trapz(HMF_array * NSAT * NSAT * U_FT * U_FT, M_h_array)
    return intg * 1 / np.power(N_G, 2)

def PS_1h(k, M_min, sigma_logM, M_sat, alpha, z,
          crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT):
    PS1cs = PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z,
                    crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT)
    PS1ss = PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z,
                     crit_dens_rescaled, M_h_array, HMF_array, N_G, NSAT)
    return PS1cs + PS1ss

def halo_bias_TINKER(nu, rc = 1.686):
    y = np.log10(200)
    A = 1.0 + 0.24 * y * np.exp(- (4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(- (4 / y) ** 4)
    c = 2.4
    _Atrm = A * np.power(nu,a) / (np.power(nu,a) + rc**a)
    _Btrm = B * np.power(nu,b)
    _Ctrm = C * np.power(nu,c)
    return 1 - _Atrm + _Btrm + _Ctrm

def PS_2h(k, M_min, sigma_logM, M_sat, alpha, z,
          crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
          bias, hmf_k, hmf_PS, D_ratio, _PS_NORM_, USE_MY_PS):
    NTOT = NCEN + NSAT
    U_FT = u_FT(k, M_h_array, z, crit_dens_rescaled)
    if (USE_MY_PS):
        PS_m = power_spectrum(k, z, D_ratio, _PS_NORM_)
        intg = np.trapz(NTOT * HMF_array * bias * U_FT, M_h_array)
        return PS_m * np.power(intg / N_G, 2)
    else:
        intg = np.trapz(NTOT * HMF_array * bias * U_FT, M_h_array)
        return np.power(intg/N_G,2)

def PS_g(k, M_min, sigma_logM, M_sat, alpha, z,
         crit_dens_rescaled, M_h_array, HMF_array, N_G,
         bias, hmf_k, hmf_PS, D_ratio, NCEN, NSAT, _PS_NORM_, USE_MY_PS):
    NCEN = N_cen(M_h_array, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    PS1 = PS_1h(k, M_min, sigma_logM, M_sat, alpha, z,
                crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT)
    PS2 = PS_2h(k, M_min, sigma_logM, M_sat, alpha, z,
                crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
                bias, hmf_k, hmf_PS, D_ratio, _PS_NORM_, USE_MY_PS)
    return PS1, PS2

###################################################################################################
def factor_k(k, theta, comoving_distance_z):
    return k / (2*np.pi) * special.j0(k * theta * comoving_distance_z)

def omega_inner_integral(theta, M_min, sigma_logM, M_sat, alpha, z, comoving_distance_z,
                        crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
                        bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS):
    if (USE_MY_PS):
        k_array = np.logspace(-3, 3, 2000)
        dlogk = np.log(k_array[1]/k_array[0])
    else:
        k_array = hmf_k
        dlogk = np.log(hmf_k[1]/hmf_k[0])
    PS_1 = [PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, \
                  crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT) for k in k_array]
    PS_2 = [PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, \
                  crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,\
                  bias, hmf_k, hmf_PS, D_ratio, _PS_NORM_, USE_MY_PS) for k in k_array]
    factor = [factor_k(k, theta, comoving_distance_z) for k in k_array]
    PS_2_corr =  np.array(PS_2) if USE_MY_PS else hmf_PS * np.array(PS_2)
    I1 = np.sum(k_array * np.array(PS_1) * factor) * dlogk
    I2 = np.sum(k_array * PS_2_corr * factor) * dlogk
    return I1, I2

def F_k_hmf_func_1(k, theta, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled, M_h_array,
                   HMF_array, N_G, NCEN, NSAT, comoving_distance_z):
    PS_1 = PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled, M_h_array,
                 HMF_array, N_G, NCEN, NSAT)
    factor = factor_k(k, theta, comoving_distance_z)
    return float(PS_1 * factor)

def F_k_hmf_func_2(k, theta, M_min, sigma_logM, M_sat, alpha, z,
                   crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
                   bias, hmf_k, hmf_PS, D_ratio, _PS_NORM_, comoving_distance_z, USE_MY_PS):
    PS_2 = PS_2h(k, M_min, sigma_logM, M_sat, alpha, z,
                 crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
                 bias, hmf_k, hmf_PS, D_ratio, _PS_NORM_, USE_MY_PS)
    PS_2_corr =  PS_2 if USE_MY_PS else hmf_PS * PS_2
    factor = factor_k(k, theta, comoving_distance_z)
    return float(PS_2_corr * factor)

def omega_inner_integral_1(theta, M_min, sigma_logM, M_sat, alpha, z,
                           comoving_distance_z, crit_dens_rescaled, M_h_array, HMF_array, N_G,
                           NCEN, NSAT, bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS):
    if (USE_MY_PS):
        k_array = np.logspace(-3.5, 3.5, 2500)
        dlogk = np.log(k_array[1]/k_array[0])
    else:
        k_array = hmf_k
        dlogk = np.log(hmf_k[1]/hmf_k[0])
    PS_1 = [PS_1h(k, M_min, sigma_logM, M_sat, alpha, z,\
                  crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT) for k in k_array]
    factor = [factor_k(k, theta, comoving_distance_z) for k in k_array]
    # return np.sum(k_array * np.array(PS_1) * factor) * dlogk
    # return np.trapz(np.array(PS_1) * factor, k_array)
    _args = (theta, M_min, sigma_logM, M_sat, alpha, z,\
             crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT, comoving_distance_z)
    return integrate.quad(F_k_hmf_func_1, 1e-3, 1e5, limit=10000, epsabs=1e-3, args = _args)[0]

def omega_inner_integral_2(theta, M_min, sigma_logM, M_sat, alpha, z,
                           comoving_distance_z, crit_dens_rescaled, M_h_array, HMF_array, N_G,
                           NCEN, NSAT, bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS):
    if (USE_MY_PS):
        k_array = np.logspace(-3.5, 3.5, 2500)
        dlogk = np.log(k_array[1]/k_array[0])
    else:
        k_array = hmf_k
        dlogk = np.log(hmf_k[1]/hmf_k[0])
    PS_2 = [PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,\
                  M_h_array, HMF_array, N_G, NCEN, NSAT, bias, hmf_k, hmf_PS,\
                  D_ratio, _PS_NORM_, USE_MY_PS) for k in k_array]
    factor = [factor_k(k, theta, comoving_distance_z) for k in k_array]
    PS_2_corr =  np.array(PS_2) if USE_MY_PS else hmf_PS * np.array(PS_2)
    # return np.sum(k_array * PS_2_corr * factor) * dlogk
    # return np.trapz(np.array(PS_2_corr) * factor, k_array)
    _args = (theta, M_min, sigma_logM, M_sat, alpha, z,\
             crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN,\
             NSAT, bias, hmf_k, hmf_PS, D_ratio, _PS_NORM_, comoving_distance_z, USE_MY_PS)
    return integrate.quad(F_k_hmf_func_2, 1e-3, 1e5, limit=10000, epsabs=1e-3, args = _args)[0]

def omega_z_component_1(z, theta, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT, _PS_NORM_,
                        USE_MY_PS = True, REWRITE_TBLS = False):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z, REWRITE_TBLS)
    D_ratio   = (D_growth_factor(z)/D_growth_factor(0))**2 if z != 0 else 1
    N_G  = n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array)
    comoving_distance_z = cosmo.comoving_distance(z).value
    crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
    return omega_inner_integral_1(theta, M_min, sigma_logM, M_sat, alpha, z, comoving_distance_z,
                                  crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
                                  nu_array, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS)

def omega_z_component_2(z, theta, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT, _PS_NORM_,
                        USE_MY_PS = True, REWRITE_TBLS = False):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z, REWRITE_TBLS)
    bias = Tinker10(nu=nu_array).bias() if 1 else halo_bias_TINKER(nu_array)
    D_ratio   = (D_growth_factor(z)/D_growth_factor(0))**2 if z != 0 else 1
    N_G  = n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array)
    comoving_distance_z = cosmo.comoving_distance(z).value
    crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
    return omega_inner_integral_2(theta, M_min, sigma_logM, M_sat, alpha, z, comoving_distance_z,
                                  crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
                                  bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS)

def omega(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
          USE_MY_PS = True, REWRITE_TBLS = False):
    _PS_NORM_ = norm_power_spectrum()
    M_h_array, _, __, ___, ____ = init_lookup_table(0, REWRITE_TBLS)
    NCEN = N_cen(M_h_array, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z =  np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    intg1 = [omega_z_component_1(z, theta, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT, _PS_NORM_,\
         USE_MY_PS = USE_MY_PS, REWRITE_TBLS = REWRITE_TBLS) for z in z_array]
    intg2 = [omega_z_component_2(z, theta, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT, _PS_NORM_,\
         USE_MY_PS = USE_MY_PS, REWRITE_TBLS = REWRITE_TBLS) for z in z_array]
    I1 = np.trapz(np.array(intg1) * factor_z, z_array)
    I2 = np.trapz(np.array(intg2) * factor_z, z_array)
    return I1, I2
def omega_array(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
                USE_MY_PS = True, REWRITE_TBLS = False):
    omega1h, omega2h = np.zeros(0), np.zeros(0)
    for tht in tqdm(theta):
        o_1h, o_2h = omega(tht, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
                           USE_MY_PS = USE_MY_PS, REWRITE_TBLS=REWRITE_TBLS)
        omega1h, omega2h = np.append(omega1h, o_1h), np.append(omega2h, o_2h)
    omega1h[omega1h<0] = 0
    omega1h[(theta/(1/206265))>50] = 0
    return omega1h, omega2h

###################################################################################################
### AVG QUANTITIES ################################################################################
def get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, z_cen, N_z_nrm):
    _N_G, _dVdz = np.zeros(0),  np.zeros(0)
    for z in z_array:
        M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
        _N_G  = np.append(_N_G, n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array))
        _dVdz = np.append(_dVdz, cosmo.comoving_distance(z).value**2 * c_light / cosmo.H(z).value)
    return np.trapz(_N_G * _dVdz * N_z_nrm, z_array)/np.trapz(_dVdz * N_z_nrm, z_array)

def get_AVG_N_tot(M_min, sigma_logM, M_sat, alpha, z, LOG_INTGR = False):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    if(LOG_INTGR):
        return np.sum(M_h_array*HMF_array*NTOT)/np.sum(M_h_array*HMF_array)
    else:
        return np.trapz(HMF_array*NTOT, M_h_array)/np.trapz(HMF_array, M_h_array)

def get_AVG_Host_Halo_Mass(M_min, sigma_logM, M_sat, alpha, z, LOG_INTGR = False):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    if(LOG_INTGR):
        return np.sum(M_h_array*M_h_array*HMF_array*NTOT)/np.sum(M_h_array*HMF_array*NTOT)
    else:
        return np.trapz(M_h_array*HMF_array*NTOT, M_h_array)/np.trapz(HMF_array*NTOT, M_h_array)

def get_EFF_gal_bias(M_min, sigma_logM, M_sat, alpha, z, LOG_INTGR = False, USE_MY_BIAS = True):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
    if USE_MY_BIAS:
        bias = halo_bias_TINKER(nu_array)
    else:
        bias = Tinker10(nu=nu_array).bias()
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    if(LOG_INTGR):
        return np.sum(bias*M_h_array*HMF_array*NTOT)/np.sum(M_h_array*HMF_array*NTOT)
    else:
        return np.trapz(bias*HMF_array*NTOT, M_h_array)/np.trapz(HMF_array*NTOT, M_h_array)

def get_AVG_f_sat(M_min, sigma_logM, M_sat, alpha, z, LOG_INTGR = False):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    if(LOG_INTGR):
        return np.sum(M_h_array*HMF_array*NSAT)/np.sum(M_h_array*HMF_array*NTOT)
    else:
        return np.trapz(HMF_array*NSAT, M_h_array)/np.trapz(HMF_array*NTOT, M_h_array)

###################################################################################################
#### INITIALIZE HMF ###############################################################################
def init_lookup_table(z, REWRITE_TBLS = False):
    _HERE_PATH = os.path.split(os.path.dirname(os.path.abspath('')))[0]
    FOLDERPATH = _HERE_PATH + '/GALESS/galess/data/HMF_tables/'
    if os.path.exists(FOLDERPATH):
        FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1, 2)*100))+'.txt'
        if (os.path.isfile(FPATH) and not REWRITE_TBLS):
            hmf_mass, hmf_dndm, hmf_nu = np.loadtxt(FPATH, delimiter=',')
            FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1, 2)*100))+'_PS.txt'
            hmf_k, hmf_PS = np.loadtxt(FPATH, delimiter=',')
        else:
            print(f'Calculating HMF table at redshift {z:.2f}')
            hmf = MassFunction(Mmin = 9, Mmax = 18, dlog10m = 0.1, lnk_min = -11.1, lnk_max = 9.6,
                               dlnk=0.002, z=z, hmf_model = "Behroozi", sigma_8 = 0.8159,
                               cosmo_params = {'Om0':OmegaM, 'H0': 100*h})
            hmf_mass = hmf.m / h
            hmf_dndm = hmf.dndm * h**4
            hmf_nu   = hmf.nu
            np.savetxt(FPATH, (hmf_mass, hmf_dndm, hmf_nu),  delimiter=',')
            rd_st = 'redshift_' + str(int(z)) + '_'+str(int(np.around(z%1, 2) * 100)) + '_PS.txt'
            FPATH = FOLDERPATH + rd_st
            hmf_k = hmf.k * h
            hmf_PS_Nln   = hmf.nonlinear_power / h**3
            # hmf_PS_Lin   = hmf.power / h**3
            # hmf_PS   = hmf_PS_Lin + hmf_PS_Nln
            hmf_PS = hmf_PS_Nln
            np.savetxt(FPATH, (hmf_k, hmf_PS),  delimiter=',')
        return hmf_mass, hmf_dndm, hmf_nu, hmf_k, hmf_PS
    else:
        print(FOLDERPATH)
        raise ValueError('Folder does not exist.')
###################################################################################################
###################################################################################################
def DEBUG_omega_inner_integral(theta, M_min, sigma_logM, M_sat, alpha, z, M_h_array,
                               HMF_array, N_G, sigma_array, D_ratio, _PS_NORM_,
                               ONLY_1H_TERM, VERBOSE):
    for k in np.array([0.0001, 0.001, 0.01, 0.1, 1, 10]):
        PS1h = PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)
        PS1c = PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)
        PS1s = PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)
        PS2n = PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G,
                    sigma_array, D_ratio, _PS_NORM_)
        print(f' --- --- k = {k}')
        print(f' --- --- PS_1h : {PS1h}')
        print(f' --- --- --- PS_cs : {PS1c}')
        print(f' --- --- --- PS_ss : {PS1s}')
        print(f' --- --- PS_2h : {PS2n}')
        NCEN = N_cen(M_h_array, M_min, sigma_logM)
        NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
        U_FT = u_FT(k, M_h_array, z)
        intg = np.trapz(HMF_array*NCEN*NSAT*U_FT, M_h_array)
        print(f' --- --- --- --- NCEN : {np.mean(NCEN)}')
        print(f' --- --- --- --- NSAT : {np.mean(NSAT)}')
        print(f' --- --- --- --- U_FT : {np.mean(U_FT)}')
        print(f' --- --- --- --- intg : {intg}')
        print(f' --- --- J Bessel : {special.j0(k * theta * cosmo.comoving_distance(z).value)}')
    pass
###################################################################################################