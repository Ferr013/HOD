import warnings
warnings.simplefilter("ignore")

import sys, os
import numpy as np
from tqdm.notebook import tqdm
from scipy import special
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import uuid
from hmf import MassFunction
from halomod.bias import Tinker10
from astropy.cosmology import Planck15
from scipy.integrate import simpson
import gzip

cosmo = Planck15 #FlatLambdaCDM(H0=67.74, Om0=0.3089, Tcmb0=2.725)
sigma_8 = 0.8159
h = cosmo.H(0).value/100
c_light  = 299792.458 #speed of light km/s

###################################################################################################
### Luminosity -> Halo Mass from Tacchella, Trenti et al. 2015 ####################################
def load_Tacchella_table():
    file_path = r"Trenti_15.dat.gz"
    z, p, mag_d, mag, Mstar, Mdm, logZ, N = [], [], [], [], [], [], [], []
    with gzip.open(file_path, 'rt') as file:
        for line in file:
            columns = line.strip().split(' ')
            z.append(float(columns[0]))
            p.append(float(columns[1]))
            mag_d.append(float(columns[2]))
            mag.append(float(columns[3]))
            Mstar.append(float(columns[4]))
            Mdm.append(float(columns[5]))
            # logZ.append(list(columns[6]))
            N.append(float(columns[7]))
    z, p, mag_d = np.array(z), np.array(p), np.array(mag_d)
    mag, Mstar, Mdm = np.array(mag), np.array(Mstar), np.array(Mdm)
    N = np.array(N) # logZ = np.array(logZ)
    return z, p, mag_d, mag, Mstar, Mdm, logZ, N

def get_M_DM_range(z_analysis=5, m_max=-15, m_min=-22, delta_z=0.5, VERBOSE=False):
    z, p, mag_d, mag, Mstar, Mdm, logZ, N = load_Tacchella_table()
    zmax, zmin = z_analysis + delta_z, z_analysis - delta_z
    if m_max < 0: #check if abs magnitudes
        _m_max,_m_min = np.max((m_max, m_min)), np.min((m_max, m_min))
        mag_max, mag_min = _m_max, _m_min
    else:
        _m_max,_m_min = np.max((m_max, m_min)), np.min((m_max, m_min))
        _distmd = 2.5 * np.log10(1+z_analysis) - cosmo.distmod(z_analysis).value
        mag_max, mag_min = _m_max + _distmd, _m_min + _distmd
    idx = np.where((z>=zmin) & (z<zmax) & (p==max(p)) & (mag<mag_max) & (mag>mag_min))[0]
    if len(idx) < 2:
        if VERBOSE: print('The redshift and/or mass interval requested are not in the lookup table')
        if z_analysis > 1:
            if VERBOSE: print('Trying z-0.5 --> z : ', z_analysis - 0.5)
            return get_M_DM_range(z_analysis - 1, m_max, m_min, delta_z)
        return -99, -99
    magg, mmdm = mag[idx], Mdm[idx]
    idx_sort   = np.argsort(magg)
    magg, mmdm = magg[idx_sort], mmdm[idx_sort]
    return np.log10(min(mmdm)), np.log10(max(mmdm))

###################################################################################################
### HALO OCCUPATION DISTRIBUTION ##################################################################
def N_cen(M_h, M_min, sigma_logM, DC = 1):
    return DC * 0.5*(1+special.erf((np.log10(M_h)-np.log10(M_min))/(np.sqrt(2)*sigma_logM)))

def N_sat(M_h, M_sat, alpha, M_min, sigma_logM, DC = 1):
    M_cut = np.power(M_min, -0.5) #Harikane 2018
    return DC * N_cen(M_h, M_min, sigma_logM) * np.power((M_h-M_cut)/M_sat,alpha)

def N_tot(M_h, M_sat, alpha, M_min, sigma_logM, DC = 1):
    return DC * (N_cen(M_h, M_min, sigma_logM) + N_sat(M_h, M_sat, alpha, M_min, sigma_logM))

def get_c_from_M_h(M_h, z, model='Correa'):
    if model == 'Correa':
        #Correa ET AL: Eq.20 https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.1217C/abstract
        alpha = 1.226 - 0.1009*(1 + z) + 0.00378*(1 + z)**2
        beta  = 0.008634 - 0.08814*np.power((1 + z), -0.58816)
        log_c = alpha + beta * np.log10(M_h)
        return np.power(10, log_c)
    elif model == 'Duffy':
        #DUFFY ET AL: Eq.4 https://arxiv.org/pdf/0804.2486.pdf
        M_pivot = 2e12/h #M_sun
        A, B, C = 6.71, -0.091, -0.44 #Relaxed
        #A, B, C = 5.71, -0.084, -0.47 #Full
        return A * np.power(M_h / M_pivot, B) * (1+z) ** C
    else:
        #Eq.4 https://iopscience.iop.org/article/10.1086/367955/pdf
        c_norm = 8
        return (c_norm) / (1+z) * np.power(M_h / (1.4e14), -0.13)

def u_FT(k, M_h, z, crit_dens_rescaled):
    r_v = np.power(M_h/crit_dens_rescaled, 1/3) #rho = M_sun/Mpc^3
    c   = get_c_from_M_h(M_h, z)
    f_c = np.log(1+c)-c/(1+c)
    r_s = r_v/c
    si, ci = special.sici(k*r_s)
    si_c, ci_c = special.sici(k*r_s*(1+c))
    return (np.sin(k*r_s)*(si_c-si)+np.cos(k*r_s)*(ci_c-ci)-(np.sin(c*k*r_s)/((1+c)*k*r_s)))/f_c

def PS_1_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias):
    PS1cs = np.trapz(HMF_array * NCEN * NSAT * U_FT, M_h_array) * 2
    PS1ss = np.trapz(HMF_array * NSAT * NSAT * U_FT * U_FT, M_h_array) * 1
    PS_2h = np.power(np.trapz((NCEN + NSAT) * HMF_array * bias * U_FT, M_h_array), 2)
    return np.array([(PS1cs + PS1ss), PS_2h])

def PS_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias):
    PS_2h = np.power(np.trapz((NCEN + NSAT) * HMF_array * bias * U_FT, M_h_array), 2)
    return PS_2h

def interpolate_between_J0_zeros(i, k0, k_array, log_hmf_PS, log_PS_1, log_PS_2, theta, comoving_distance_z,
                                 N_INTRP = 8, STEP_J0 = 50_000):
    mask_k = np.logical_and(k_array> k0[i], k_array< k0[i+1])
    j = 1
    while not np.any(mask_k) and (i+j)<STEP_J0 and j<1000:
        mask_k = np.logical_and(k_array> k0[i], k_array< k0[i+1+j])
        j += 1
    if np.any(mask_k):
        iks, ikb = np.where(mask_k)[0][0]-1, np.where(mask_k)[0][-1]+1
        if iks>0 and ikb<len(k_array):
            log_k_array = np.log10(k_array)
            log_k_upper = np.log10(k0[i+1])
            aug_log_k_itv = np.log10(np.logspace(log_k_array[iks], log_k_upper, N_INTRP))
            log_dk = log_k_array[ikb] - log_k_array[iks]
            dv_logPS = (log_hmf_PS[ikb] - log_hmf_PS[iks])/log_dk
            dv_logP1 = (log_PS_1[ikb] - log_PS_1[iks])/log_dk
            dv_logP2 = (log_PS_2[ikb] - log_PS_2[iks])/log_dk
            # print(log_dk, dv_logPS)
            aug_PS = np.power(10, log_hmf_PS[iks] + dv_logPS * (aug_log_k_itv - log_k_array[iks]))
            aug_P1 = np.power(10, log_PS_1[iks] + dv_logP1 * (aug_log_k_itv - log_k_array[iks]))
            aug_P2 = np.power(10, log_PS_2[iks] + dv_logP2 * (aug_log_k_itv - log_k_array[iks]))
            aug_k_itv = np.power(10, aug_log_k_itv)
            # print((aug_log_k_itv - log_k_array[iks]))
            aug_J0 = np.array([special.j0(k*theta*comoving_distance_z) for k in aug_k_itv])
            # mask_augk = np.logical_and(aug_k_itv> k0[i], aug_k_itv< k0[i+1])
            # aP1, aP2 = aug_P1[mask_augk], aug_P2[mask_augk]
            # aPS, aJ0 = aug_PS[mask_augk], aug_J0[mask_augk]
            # ak = np.append(k0[i], np.append(aug_k_itv[mask_augk], k0[i+1]))
            # print(log_k_array[iks], log_k_array[ikb])
            # print('-> len(mask)', np.sum(mask_k), ' ', j,' !!! ', aug_P1, aug_P2, aug_PS, aug_J0, aug_k_itv)
            return aug_P1[1:-1], aug_P2[1:-1], aug_PS[1:-1], aug_J0[1:-1], aug_k_itv
    else:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

def integral_P1gsJzero_from_largek(k_min, z, comoving_distance_z, theta_rad,
                                   VERBOSE = False):
    # r = comoving_distance_z
    # theta = theta_arcsec/206265
    a, b, c = -0.04346161,  0.13011286,  1.35990887 # from P_1cs fit
    A = np.power(10, a * z*z + b * z + c)
    y = k_min/(comoving_distance_z*theta_rad)
    pre = comoving_distance_z*theta_rad/(2*np.pi)/2
    fir = (-np.pi*y**2*special.struve(1, y)+2*y*y+2)*special.jv(0, y)/y
    sec = (np.pi*y*special.struve(0, y)-2)*special.jv(1, y)-2
    if VERBOSE:
        print(f' - y: {y:.1e}')
        print(f' --- pre: {pre:.1e}')
        print(f' --- fir: {fir:.1e}')
        print(f' --- sec: {sec:.1e}')
    return A * pre * (fir + sec)

def integrate_between_J_zeros(theta, z, comoving_distance_z,
                              _k_array, hmf_PS, PS_1, PS_2,
                              STEP_J0 = 50_000, PRECISION = 0.01,
                              INTERPOLATION = False,
                              VERBOSE = False):
    k_array = 0
    APPROX_LARGE_K = 0
    if APPROX_LARGE_K:
        K_MIN = 1e4
        mk = _k_array<=K_MIN
        low_k_array = _k_array[mk]
        PS_1, PS_2, hmf_PS = PS_1[mk], PS_2[mk], hmf_PS[mk]
        k_array = low_k_array
    else:
        k_array = _k_array

    j_0_zeros = np.append(1e-4, special.jn_zeros(0, STEP_J0))
    res1, res2 = np.zeros(0), np.zeros(0)
    k0 = j_0_zeros / theta / comoving_distance_z
    mask_k = k_array < k0[0]
    k_intr = np.append(k_array[mask_k], k0[0])
    Bessel = np.array([special.j0(k*theta*comoving_distance_z) for k in k_array[mask_k]])
    int1 = simpson(np.append(PS_1[mask_k], 0)
                    * k_intr / (2*np.pi) * np.append(Bessel, 0), k_intr)
    int2 = simpson(np.append(hmf_PS[mask_k], 0) * np.append(PS_2[mask_k],0)
                    * k_intr / (2*np.pi) * np.append(Bessel, 0), k_intr)
    res1 = np.append(res1, int1)
    res2 = np.append(res2, int2)
    i, FLAG_DENSITY = 0, True
    r1h, r2h = 0, 0
    e1h, e2h = np.inf, np.inf

    while (e1h > PRECISION or e2h > PRECISION) and (i < STEP_J0-1) and FLAG_DENSITY:
        mask_k = np.logical_and(k_array> k0[i], k_array< k0[i+1])
        k_intr = np.append(k0[i], np.append(k_array[mask_k], k0[i+1]))
        aPS, aP1, aP2, aJ0, ak = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        if len(k_array[mask_k]) < 3 and INTERPOLATION:
            aPS, aP1, aP2, aJ0, ak = interpolate_between_J0_zeros(i, k0, k_array,
                                                                  np.log10(hmf_PS),
                                                                  np.log10(PS_1),
                                                                  np.log10(PS_2),
                                                                  theta, comoving_distance_z,
                                                                  N_INTRP = 100, STEP_J0 = 20_000)
            # return 0, 0, 0, 0
        else:
            Bessel = np.array([special.j0(k*theta*comoving_distance_z) for k in k_array[mask_k]])
            aPS, aP1, aP2, aJ0, ak = hmf_PS[mask_k], PS_1[mask_k], PS_2[mask_k], Bessel, k_intr
        if len(aPS) == 0:
            # print('len(aPS) == 0 :', len(aPS) == 0)
            # print('len(k_array[mask_k]):', len(k_array[mask_k]))
            # print('(e1h > 0.05) : ', e1h > 0.05)
            # print('(e2h > 0.05) : ', e2h > 0.05)
            # print('(i < STEP_J0-1) :', (i < STEP_J0-1))
            FLAG_DENSITY = False
        else:
            int1 = simpson(np.pad(aP1, 1)
                            * ak / (2*np.pi) * np.pad(aJ0, 1), ak)
            int2 = simpson(np.pad(aPS, 1) * np.pad(aP2,1)
                            * ak / (2*np.pi) * np.pad(aJ0, 1), ak)
            res1, res2 = np.append(res1, int1), np.append(res2, int2)
            e1h = np.abs((np.sum(res1) - r1h)/np.sum(res1))
            e2h = np.abs((np.sum(res2) - r2h)/np.sum(res2))
            r1h, r2h = np.sum(res1), np.sum(res2)
            i += 1
    if 0: #VERBOSE:
        print(f'### z: {z} ###')
        print(f'##### theta: {theta*206265} arcsec #####')
        print(f'Integrating over {i} zeros of Bessel J0')
        k_min = np.log10(np.min(k_array))
        k_max_J0 = np.log10(special.jn_zeros(0, STEP_J0)[i]/ theta / comoving_distance_z)
        print(f'Over a range of {k_min:.1e} < k < {k_max_J0:.1e}')
        print(f'     Integral 1h: {r1h:.1e} \pm {e1h*100:.1e}%')
        print(f'     Integral 2h: {r2h:.1e} \pm {e2h*100:.1e}%')
    if APPROX_LARGE_K:
        r1h = r1h + integral_P1gsJzero_from_largek(K_MIN, z, comoving_distance_z, theta)
    return r1h, r2h, e1h, e2h

def integrate_between_J_zeros_2halo(theta, z, comoving_distance_z,
                                    _k_array, hmf_PS, PS_2,
                                    STEP_J0 = 50_000, PRECISION = 0.01,
                                    INTERPOLATION = False,
                                    VERBOSE = False):
    k_array = 0
    APPROX_LARGE_K = 0
    if APPROX_LARGE_K:
        K_MIN = 1e4
        mk = _k_array<=K_MIN
        low_k_array = _k_array[mk]
        PS_1, PS_2, hmf_PS = PS_1[mk], PS_2[mk], hmf_PS[mk]
        k_array = low_k_array
    else:
        k_array = _k_array

    j_0_zeros = np.append(1e-4, special.jn_zeros(0, STEP_J0))
    res2 = np.zeros(0)
    k0 = j_0_zeros / theta / comoving_distance_z
    mask_k = k_array < k0[0]
    k_intr = np.append(k_array[mask_k], k0[0])
    Bessel = np.array([special.j0(k*theta*comoving_distance_z) for k in k_array[mask_k]])
    int2 = simpson(np.append(hmf_PS[mask_k], 0) * np.append(PS_2[mask_k],0)
                    * k_intr / (2*np.pi) * np.append(Bessel, 0), k_intr)
    res2 = np.append(res2, int2)
    i, FLAG_DENSITY = 0, True
    r2h, e2h = 0, np.inf
    while (e2h > PRECISION) and (i < STEP_J0-1) and FLAG_DENSITY:
        mask_k = np.logical_and(k_array> k0[i], k_array< k0[i+1])
        k_intr = np.append(k0[i], np.append(k_array[mask_k], k0[i+1]))
        aPS, aP2, aJ0, ak = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        Bessel = np.array([special.j0(k*theta*comoving_distance_z) for k in k_array[mask_k]])
        aPS, aP2, aJ0, ak = hmf_PS[mask_k], PS_2[mask_k], Bessel, k_intr
        if len(aPS) == 0: FLAG_DENSITY = False
        else:
            int2 = simpson(np.pad(aPS, 1) * np.pad(aP2,1)
                            * ak / (2*np.pi) * np.pad(aJ0, 1), ak)
            res2 = np.append(res2, int2)
            e2h  = np.abs((np.sum(res2) - r2h)/np.sum(res2))
            r2h  =  np.sum(res2)
            i += 1
    return r2h, e2h

def omega_inner_integral(theta, z, comoving_distance_z, M_h_array, HMF_array, NCEN, NSAT, U_FT,
                         k_array, hmf_PS, bias, STEP_J0 = 50_000, INTERPOLATION = True, VERBOSE = False):
    PS_1, PS_2 = PS_1_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias)
    N1, N2 = np.max(PS_1), np.max(PS_2)
    PS_1, PS_2 = PS_1 / N1, PS_2 / N2
    if 1:
        PRECISION = 0.01
        R_T = np.asarray([integrate_between_J_zeros(t, z, comoving_distance_z, k_array,
                                                    hmf_PS, PS_1, PS_2, STEP_J0,
                                                    PRECISION, INTERPOLATION, VERBOSE)
                                                    for t in theta])
        R_T1, R_T2 = R_T.T[0], R_T.T[1]
        E_T1, E_T2 = R_T.T[2], R_T.T[3]
        return R_T1 * N1, R_T2 * N2, E_T1, E_T2
    Bessel = np.array([\
             np.array([special.j0(k*t*comoving_distance_z) for k in k_array])\
             for t in theta])
    R_T1 = np.trapz(PS_1 * k_array / (2*np.pi) * Bessel, k_array, axis = -1)
    R_T2 = np.trapz(hmf_PS * PS_2 * k_array / (2*np.pi) * Bessel, k_array, axis = -1)
    return R_T1 * N1, R_T2 * N2, 0, 0

def omega_inner_integral_2halo(theta, z, comoving_distance_z, M_h_array, HMF_array, NCEN, NSAT, U_FT,
                         k_array, hmf_PS, bias, STEP_J0 = 50_000, INTERPOLATION = True, VERBOSE = False):
    PS_2 = PS_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias)
    N2   = np.max(PS_2)
    PS_2 = PS_2 / N2
    if 1:
        PRECISION = 0.01
        R_T = np.asarray([integrate_between_J_zeros_2halo(t, z, comoving_distance_z, k_array,
                                                            hmf_PS, PS_2, STEP_J0,
                                                            PRECISION, INTERPOLATION, VERBOSE)
                                                            for t in theta])
        R_T2, E_T2 = R_T.T[0],  R_T.T[1]
        return R_T2 * N2, E_T2
    Bessel = np.array([\
             np.array([special.j0(k*t*comoving_distance_z) for k in k_array])\
             for t in theta])
    R_T2 = np.trapz(hmf_PS * PS_2 * k_array / (2*np.pi) * Bessel, k_array, axis = -1)
    return R_T2 * N2, 0, 0

def init(mem):
    global mem_id
    mem_id = mem
    return

def omega_z_component_single(args):
    job_id, shape, z, _args_ = args
    theta, M_DM_min, M_DM_max, NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, STEP_J0, INTERPOLATION, VERBOSE = _args_
    shmem = SharedMemory(name=f'{mem_id}', create=False)
    shres = np.ndarray(shape, buffer=shmem.buf, dtype=np.float64)
    if PRECOMP_UFT:
        M_h_array, HMF_array, nu_array, k_array, hmf_PS, U_FT =\
            init_lookup_table(z, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
    else:
        M_h_array, HMF_array, nu_array, k_array, hmf_PS = \
            init_lookup_table(z, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
        crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
        U_FT = np.array([u_FT(k, M_h_array, z, crit_dens_rescaled) for k in k_array])

    if VERBOSE: print('len (M_h_array) @ z = ',z,' : ', len(M_h_array))
    bias = Tinker10(nu=nu_array, sigma_8 = sigma_8, cosmo = cosmo).bias()
    comoving_distance_z = cosmo.comoving_distance(z).value
    oz1, oz2, e1, e2 = omega_inner_integral(theta, z, comoving_distance_z, M_h_array, HMF_array,
                                            NCEN, NSAT, U_FT, k_array, hmf_PS, bias,
                                            STEP_J0, INTERPOLATION, VERBOSE)
    shres[job_id, 0, :], shres[job_id, 1, :] = oz1, oz2
    if 0:
        print(f'Integrating over {STEP_J0} zeros of Bessel J0')
        print(f'### z: {z} ###')
        for it, _theta in enumerate(theta):
            print(f'##### theta: {_theta*206265} arcsec ###')
            print(f'     Integral 1h: {oz1[it]:.1e} \pm {e1[it]*100:.1e}%')
            print(f'     Integral 2h: {oz2[it]:.1e} \pm {e2[it]*100:.1e}%')
            k_min = np.log10(np.min(k_array))
            k_max_J0 = np.log10(special.jn_zeros(0, STEP_J0)[-1]/ _theta / comoving_distance_z)
            print(f'Over a range of {k_min:.1e} < k < {k_max_J0:.1e}')
    return

def omega_z_component_single_2halo(args):
    job_id, shape, z, _args_ = args
    theta, M_DM_min, M_DM_max, NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, STEP_J0, INTERPOLATION, VERBOSE = _args_
    shmem = SharedMemory(name=f'{mem_id}', create=False)
    shres = np.ndarray(shape, buffer=shmem.buf, dtype=np.float64)
    if PRECOMP_UFT:
        M_h_array, HMF_array, nu_array, k_array, hmf_PS, U_FT =\
            init_lookup_table(z, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
    else:
        M_h_array, HMF_array, nu_array, k_array, hmf_PS = \
            init_lookup_table(z, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
        crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
        U_FT = np.array([u_FT(k, M_h_array, z, crit_dens_rescaled) for k in k_array])

    if VERBOSE: print('len (M_h_array) @ z = ',z,' : ', len(M_h_array))
    bias = Tinker10(nu=nu_array, sigma_8 = sigma_8, cosmo = cosmo).bias()
    comoving_distance_z = cosmo.comoving_distance(z).value
    oz2, e2 = omega_inner_integral_2halo(theta, z, comoving_distance_z, M_h_array, HMF_array,
                                        NCEN, NSAT, U_FT, k_array, hmf_PS, bias,
                                        STEP_J0, INTERPOLATION, VERBOSE)
    shres[job_id, :] = oz2
    if VERBOSE:
        print(f'Integrating over {STEP_J0} zeros of Bessel J0')
        print(f'### z: {z} ###')
        for it, _theta in enumerate(theta):
            print(f'##### theta: {_theta*206265} arcsec ###')
            print(f'     Integral 2h: {oz2[it]:.1e} \pm {e2[it]*100:.1e}%')
            k_min = np.log10(np.min(k_array))
            k_max_J0 = np.log10(special.jn_zeros(0, STEP_J0)[-1]/ _theta / comoving_distance_z)
            print(f'Over a range of {k_min:.1e} < k < {k_max_J0:.1e}')
    return

def omega_z_component_parallel(z_array, theta_array, M_DM_min, M_DM_max, NCEN, NSAT,
                               PRECOMP_UFT = False, REWRITE_TBLS = False,
                               LOW_RES = False, STEP_J0 = 50_000,
                               cores=None, INTERPOLATION = True, VERBOSE = False):
    if cores is None:
        if len(z_array) <= multiprocessing.cpu_count():
            cores = len(z_array)
        else:
            print('MORE Z BINS THAN CORES, THE CODE IS NOT SMART ENOUGH TO HANDLE THIS YET')
            raise ValueError
            #cores = multiprocessing.cpu_count()
    _args_ = theta_array, M_DM_min, M_DM_max, NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS,\
             LOW_RES, STEP_J0, INTERPOLATION, VERBOSE
    shape = (len(z_array), 2, len(theta_array))
    args =  [(i, shape, z_array[i], _args_) for i in range(int(cores))]
    exit = False
    try:
        global mem_id
        mem_id = str(uuid.uuid1())[:30] #Avoid OSError: [Errno 63]
        nbytes = (2 * len(z_array) * len(theta_array)) * np.float64(1).nbytes
        shd_mem = SharedMemory(name=f'{mem_id}', create=True, size=nbytes)
        method = 'spawn'
        if sys.platform.startswith('linux'):
            method = 'fork'
        ctx = multiprocessing.get_context(method)
        pool = ctx.Pool(processes=cores, maxtasksperchild=1,
                        initializer=init, initargs=(mem_id,))
        try:
            pool.map_async(omega_z_component_single, args, chunksize=1).get(timeout=10_000)
        except KeyboardInterrupt:
            print("Caught kbd interrupt")
            pool.close()
            exit = True
        else:
            pool.close()
            pool.join()
            z_h_t_array = np.ndarray(shape, buffer=shd_mem.buf, dtype=np.float64).copy()
    finally:
        shd_mem.close()
        shd_mem.unlink()
        if exit:
            sys.exit(1)
    return z_h_t_array

def omega_z_component_parallel_2halo(z_array, theta_array, M_DM_min, M_DM_max, NCEN, NSAT,
                                    PRECOMP_UFT = False, REWRITE_TBLS = False,
                                    LOW_RES = False, STEP_J0 = 50_000,
                                    cores=None, INTERPOLATION = True, VERBOSE = False):
    if cores is None:
        if len(z_array) <= multiprocessing.cpu_count():
            cores = len(z_array)
        else:
            print('MORE Z BINS THAN CORES, THE CODE IS NOT SMART ENOUGH TO HANDLE THIS YET')
            raise ValueError
            #cores = multiprocessing.cpu_count()
    cores = 1
    _args_ = theta_array, M_DM_min, M_DM_max, NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS,\
             LOW_RES, STEP_J0, INTERPOLATION, VERBOSE
    shape = (len(z_array), len(theta_array))
    args =  [(i, shape, z_array[i], _args_) for i in range(int(cores))]
    exit = False
    try:
        global mem_id
        mem_id = str(uuid.uuid1())[:30] #Avoid OSError: [Errno 63]
        nbytes = (len(z_array) * len(theta_array)) * np.float64(1).nbytes
        shd_mem = SharedMemory(name=f'{mem_id}', create=True, size=nbytes)
        method = 'spawn'
        if sys.platform.startswith('linux'):
            method = 'fork'
        ctx = multiprocessing.get_context(method)
        pool = ctx.Pool(processes=cores, maxtasksperchild=1,
                        initializer=init, initargs=(mem_id,))
        try:
            pool.map_async(omega_z_component_single_2halo, args, chunksize=1).get(timeout=10_000)
        except KeyboardInterrupt:
            print("Caught kbd interrupt")
            pool.close()
            exit = True
        else:
            pool.close()
            pool.join()
            z_h_t_array = np.ndarray(shape, buffer=shd_mem.buf, dtype=np.float64).copy()
    finally:
        shd_mem.close()
        shd_mem.unlink()
        if exit:
            sys.exit(1)
    return z_h_t_array

def omega(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
          mag_min = 0, mag_max = np.inf,
          PRECOMP_UFT = False, REWRITE_TBLS = False,
          LOW_RES = False, STEP_J0 = 50_000, cores=None,
          INTERPOLATION = False, VERBOSE = False):
    if mag_min == 0 or mag_max == np.inf:
        M_DM_min, M_DM_max = 0, np.inf
    else:
        M_DM_min, M_DM_max = get_M_DM_range(np.mean(z_array), mag_max, mag_min, delta_z=0.5)
    if VERBOSE: print('M_DM_min, M_DM_max = ', M_DM_min, M_DM_max)
    if PRECOMP_UFT:
        M_h_array, __, ___, ____, _____, _______ =\
             init_lookup_table(0, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
    else:
        M_h_array, __, ___, ____, _____ =\
             init_lookup_table(0, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
    if VERBOSE: print('len (M_h_array) @ z = 0 : ', len(M_h_array))
    NCEN = N_cen(M_h_array, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z = np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    itg = omega_z_component_parallel(z_array, theta, M_DM_min, M_DM_max, NCEN, NSAT,
                                     PRECOMP_UFT, REWRITE_TBLS,
                                     LOW_RES, STEP_J0, cores, INTERPOLATION, VERBOSE)
    #TODO: this calls init_lookuptable again, should distirbute it in (or as in) the omega_z_component_parallel
    N_G = get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm, LOW_RES)
    I1 = np.array([np.trapz(itg[:,0,i] * factor_z, z_array) for i in range(len(theta))])
    I2 = np.array([np.trapz(itg[:,1,i] * factor_z, z_array) for i in range(len(theta))])
    return I1/ np.power(N_G, 2), I2/ np.power(N_G, 2)

def omega_2halo(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
                mag_min = 0, mag_max = np.inf,
                PRECOMP_UFT = False, REWRITE_TBLS = False,
                LOW_RES = False, STEP_J0 = 50_000, cores=None,
                INTERPOLATION = False, VERBOSE = False):
    if mag_min == 0 or mag_max == np.inf:
        M_DM_min, M_DM_max = 0, np.inf
    else:
        M_DM_min, M_DM_max = get_M_DM_range(np.mean(z_array), mag_max, mag_min, delta_z=0.5)
    if VERBOSE: print('M_DM_min, M_DM_max = ', M_DM_min, M_DM_max)
    if PRECOMP_UFT:
        M_h_array, __, ___, ____, _____, _______ =\
             init_lookup_table(0, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
    else:
        M_h_array, __, ___, ____, _____ =\
             init_lookup_table(0, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
    if VERBOSE: print('len (M_h_array) @ z = 0 : ', len(M_h_array))
    NCEN = N_cen(M_h_array, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z = np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    itg = omega_z_component_parallel_2halo(z_array, theta, M_DM_min, M_DM_max, NCEN, NSAT,
                                            PRECOMP_UFT, REWRITE_TBLS,
                                            LOW_RES, STEP_J0, cores, INTERPOLATION, VERBOSE)
    #TODO: this calls init_lookuptable again, should distirbute it in (or as in) the omega_z_component_parallel
    N_G = get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
                         LOW_RES = LOW_RES, int_M_min=np.power(10, M_DM_min), int_M_max=np.power(10, M_DM_max))
    I2 = np.array([np.trapz(itg[:,i] * factor_z, z_array) for i in range(len(theta))])
    return I2/ np.power(N_G, 2)

###################################################################################################
#### INITIALIZE HMF ###############################################################################
def init_lookup_table(z, PRECOMP_UFT = False, REWRITE_TBLS = False, LOW_RES = False,
                      M_DM_min = 0, M_DM_max = np.inf):
    _HERE_PATH = os.path.dirname(os.path.abspath(''))
    FOLDERPATH = _HERE_PATH + '/HOD/HMF_tables/'
    min_lnk, max_ln_k, step_lnk = -11.5, 13.6, 0.0001
    if LOW_RES:
        FOLDERPATH = _HERE_PATH + '/HOD/HMF_tables/LowRes/'
        min_lnk, max_ln_k, step_lnk = -11.5, 13.6, 0.002
    if os.path.exists(FOLDERPATH):
        FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'.txt'
        if (os.path.isfile(FPATH) and not REWRITE_TBLS):
            hmf_mass, hmf_dndm, hmf_nu = np.loadtxt(FPATH, delimiter=',')
            FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'_PS.txt'
            hmf_k, hmf_PS = np.loadtxt(FPATH, delimiter=',')
            if PRECOMP_UFT:
                FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'_U.txt'
                hmf_U_FT = np.loadtxt(FPATH, delimiter=',')
        else:
            print(f'Calculating HMF table at redshift {z:.2f}')
            hmf = MassFunction(Mmin = 9, Mmax = 17, dlog10m = 0.025,
                               lnk_min = min_lnk, lnk_max = max_ln_k, dlnk=step_lnk,
                               z=z, hmf_model = "Behroozi", sigma_8 = sigma_8, cosmo_model = cosmo)
            hmf_mass = hmf.m / h
            hmf_dndm = hmf.dndm * h**4
            hmf_nu   = hmf.nu
            np.savetxt(FPATH, (hmf_mass, hmf_dndm, hmf_nu),  delimiter=',')
            rd_st = 'redshift_' + str(int(z)) + '_'+str(int(np.around(z%1, 2) * 100)) + '_PS.txt'
            FPATH = FOLDERPATH + rd_st
            hmf_k = hmf.k * h
            hmf_PS_Nln = hmf.nonlinear_power / h**3
            hmf_PS = hmf_PS_Nln
            np.savetxt(FPATH, (hmf_k, hmf_PS),  delimiter=',')
            FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'_U.txt'
            if PRECOMP_UFT:
                crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
                hmf_U_FT = np.array([u_FT(k, hmf_mass, z, crit_dens_rescaled) for k in hmf_k])
                np.savetxt(FPATH, hmf_U_FT,  delimiter=',')
        m_mask = np.logical_and(np.log10(hmf_mass) > M_DM_min, np.log10(hmf_mass) < M_DM_max)
        if PRECOMP_UFT:
            return hmf_mass[m_mask], hmf_dndm[m_mask], hmf_nu[m_mask], hmf_k, hmf_PS, hmf_U_FT
        else:
            return hmf_mass[m_mask], hmf_dndm[m_mask], hmf_nu[m_mask], hmf_k, hmf_PS
    else:
        print(FOLDERPATH)
        raise ValueError('Folder does not exist.')
###################################################################################################
### AVG QUANTITIES ################################################################################
def gal_density_n_g(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, DC = 1):
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM, DC)
    return np.trapz(HMF_array*NTOT, M_h_array)

def get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
                   LOW_RES = True, DC = 1, int_M_min=0, int_M_max=np.inf):
    _N_G, _dVdz = np.zeros(0),  np.zeros(0)
    for z in z_array:
        M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z, False, False, LOW_RES)
        m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
        _N_G  = np.append(_N_G, gal_density_n_g(M_min, sigma_logM, M_sat, alpha,
                                                M_h_array[m_mask], HMF_array[m_mask], DC))
        _dVdz = np.append(_dVdz, cosmo.comoving_distance(z).value**2 * c_light / cosmo.H(z).value)
    return np.trapz(_N_G * _dVdz * N_z_nrm, z_array)/np.trapz(_dVdz * N_z_nrm, z_array)

def get_AVG_N_tot(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array,
                  n_g=None, int_M_min=0, int_M_max=np.inf):
    m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
    NTOT = N_tot(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])/n_g
    return np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])/\
            np.trapz(HMF_array[m_mask], M_h_array[m_mask])

def get_AVG_Host_Halo_Mass(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array,
                           n_g=None, int_M_min=0, int_M_max=np.inf):
    m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
    NTOT = N_tot(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(M_h_array[m_mask]*HMF_array[m_mask]*NTOT, M_h_array[m_mask])/n_g
    return np.trapz(M_h_array[m_mask]*HMF_array[m_mask]*NTOT, M_h_array[m_mask])\
            /np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])

def get_EFF_gal_bias(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, nu_array,
                     n_g=None, int_M_min=0, int_M_max=np.inf):
    m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
    bias = Tinker10(nu=nu_array[m_mask], sigma_8 = sigma_8, cosmo = cosmo).bias()
    NTOT = N_tot(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(bias*HMF_array[m_mask]*NTOT, M_h_array[m_mask])/n_g
    return np.trapz(bias*HMF_array[m_mask]*NTOT, M_h_array[m_mask])/\
            np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])

def get_AVG_f_sat(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array,
                  n_g=None, int_M_min=0, int_M_max=np.inf):
    m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
    NTOT = N_tot(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    NSAT = N_sat(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(HMF_array[m_mask]*NSAT, M_h_array[m_mask])/n_g
    return np.trapz(HMF_array[m_mask]*NSAT, M_h_array[m_mask])/\
            np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])
###################################################################################################
