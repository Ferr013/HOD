import sys
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

sys.path.append(".")
import HOD

def PS_1h(k, M_min, sigma_logM, M_sat, alpha, z,
          crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT):
    PS1cs = HOD.PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z,
                    crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT)
    PS1ss = HOD.PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z,
                     crit_dens_rescaled, M_h_array, HMF_array, N_G, NSAT)
    return PS1cs + PS1ss

def PS_2h(k, M_min, sigma_logM, M_sat, alpha, z,
          crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
          bias, hmf_k, hmf_PS, D_ratio, _PS_NORM_, USE_MY_PS):
    NTOT = NCEN + NSAT
    U_FT = HOD.u_FT(k, M_h_array, z, crit_dens_rescaled)
    PS_m = HOD.power_spectrum(k, z, D_ratio, _PS_NORM_)
    intg = np.trapz(NTOT * HMF_array * bias * U_FT, M_h_array)
    return PS_m * np.power(intg / N_G, 2)

def omega_inner_integral_1(theta, M_min, sigma_logM, M_sat, alpha, z,
                           comoving_distance_z, crit_dens_rescaled, M_h_array, HMF_array, N_G,
                           NCEN, NSAT, k_array, dlogk):
    PS_1 = np.array([PS_1h(k, M_min, sigma_logM, M_sat, alpha, z,\
                crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT)*\
                k/(2*np.pi)*special.j0(k*1/206265*comoving_distance_z) for k in k_array])
    corr = np.array([np.array([special.j0(k*t*comoving_distance_z)/\
                               special.j0(k*1/206265*comoving_distance_z)\
                               for k in k_array]) for t in theta])
    R_T1 = np.sum(k_array * PS_1 * corr, axis = -1) * dlogk
    return R_T1

def omega_inner_integral_2(theta, M_min, sigma_logM, M_sat, alpha, z,
                           comoving_distance_z, crit_dens_rescaled, M_h_array, HMF_array, N_G,
                           NCEN, NSAT, k_array, dlogk, bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS):
    PS_2 = np.array([PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,\
                    M_h_array, HMF_array, N_G, NCEN, NSAT, bias, hmf_k, hmf_PS,\
                    D_ratio, _PS_NORM_, USE_MY_PS) *\
                    k/(2*np.pi)*special.j0(k*1/206265*comoving_distance_z) for k in k_array])
    corr = np.array([np.array([special.j0(k*t*comoving_distance_z)/\
                               special.j0(k*1/206265*comoving_distance_z)\
                               for k in k_array]) for t in theta])
    R_T2 = np.sum(k_array * PS_2 * corr, axis = -1) * dlogk
    return R_T2

def omega_z_component_1(z, theta, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT, _PS_NORM_, k_array, dlogk,
                        USE_MY_PS = True, REWRITE_TBLS = False):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = HOD.init_lookup_table(z, REWRITE_TBLS)
    D_ratio = (HOD.D_growth_factor(z)/HOD.D_growth_factor(0))**2 if z != 0 else 1
    N_G  = HOD.n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array)
    comoving_distance_z = cosmo.comoving_distance(z).value
    crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
    return omega_inner_integral_1(theta, M_min, sigma_logM, M_sat, alpha, z, comoving_distance_z,
                                  crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
                                  k_array, dlogk)

def omega_z_component_2(z, theta, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT, _PS_NORM_,k_array, dlogk,
                        USE_MY_PS = True, REWRITE_TBLS = False, USE_MY_BIAS = False):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = HOD.init_lookup_table(z, REWRITE_TBLS)
    bias = Tinker10(nu=nu_array).bias()
    D_ratio = (HOD.D_growth_factor(z)/HOD.D_growth_factor(0))**2 if z != 0 else 1
    N_G  = HOD.n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array)
    comoving_distance_z = cosmo.comoving_distance(z).value
    crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
    return omega_inner_integral_2(theta, M_min, sigma_logM, M_sat, alpha, z, comoving_distance_z,
                                  crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT,
                                  k_array, dlogk, bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS)

def omega(ttt, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array):
    k_array = np.logspace(-5, 4, 10000)
    dlogk = np.log(k_array[1]/k_array[0])

    _PS_NORM_ = HOD.norm_power_spectrum()
    M_h_array, _, __, ___, ____ = HOD.init_lookup_table(0)
    NCEN = HOD.N_cen(M_h_array, M_min, sigma_logM)
    NSAT = HOD.N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z =  np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    intg1 = np.array([omega_z_component_1(z, ttt, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT, _PS_NORM_, k_array, dlogk) for z in z_array])
    intg2 = np.array([omega_z_component_2(z, ttt, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT, _PS_NORM_, k_array, dlogk) for z in z_array])
    I1 = np.array([np.trapz(intg1.T[i] * factor_z, z_array) for i in range(len(ttt))])
    I2 = np.array([np.trapz(intg2.T[i] * factor_z, z_array) for i in range(len(ttt))])
    return I1, I2