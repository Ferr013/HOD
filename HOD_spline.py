################################################################################################################
# Giovanni Ferrami July 2024
################################################################################################################

import warnings
warnings.simplefilter("ignore")

import sys, os
import numpy as np
from tqdm.notebook import tqdm
from scipy import special
from scipy.interpolate import splrep, splev, splint
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

def PS_1h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias):
    PS1cs = np.trapz(HMF_array * NCEN * NSAT * U_FT, M_h_array) * 2
    PS1ss = np.trapz(HMF_array * NSAT * NSAT * U_FT * U_FT, M_h_array) * 1
    return PS1cs + PS1ss

def PS_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias):
    PS_2h = np.power(np.trapz((NCEN + NSAT) * HMF_array * bias * U_FT, M_h_array), 2)
    return PS_2h

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

def omega_inner_integral_1halo(theta, z, comoving_distance_z, M_h_array, HMF_array, NCEN, NSAT, U_FT,
                         k_array, hmf_PS, bias, STEP_J0 = 100_000, INTERPOLATION = True, VERBOSE = False):
    SPL_ORDER = 1
    PS_1 = PS_1h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias)
    N1   = np.max(PS_1)
    PS_1 = PS_1 / N1
    PS_1_spl = splrep(k_array, PS_1  , s=0, k=SPL_ORDER)
    R_T1 = np.zeros(len(theta))
    j_0_zeros = special.jn_zeros(0, STEP_J0+1)
    for it, t in enumerate(theta):
        k0 = j_0_zeros[0]/t/comoving_distance_z
        k_here = np.append(k_array[k_array<k0], k0)
        PS_1_here = splev(k_here, PS_1_spl)
        Bessel = np.array([special.j0(k*t*comoving_distance_z) for k in k_here])
        integrand = PS_1_here * k_here / (2*np.pi) * Bessel
        A_sp = splrep(k_here, integrand, s=0, k=SPL_ORDER)
        R_T1[it] += splint(k_here[0], k_here[-1], A_sp)
        i, DELTA_J0, RES_J0 = 0, 10_000, 8
        while i <= STEP_J0 - DELTA_J0:
            j_array = np.linspace(j_0_zeros[i], j_0_zeros[i+DELTA_J0], DELTA_J0*RES_J0)
            k_here = j_array / t / comoving_distance_z
            PS_1_here = splev(k_here, PS_1_spl)
            Bessel = np.array([special.j0(k*t*comoving_distance_z) for k in k_here])
            integrand = PS_1_here * k_here / (2*np.pi) * Bessel
            A_sp = splrep(k_here, integrand, s=0, k=SPL_ORDER)
            R_T1[it] += splint(k_here[0], k_here[-1], A_sp)
            i += DELTA_J0
    return R_T1 * N1, 0

def omega_inner_integral_2halo(theta, z, comoving_distance_z, M_h_array, HMF_array, NCEN, NSAT, U_FT,
                         k_array, hmf_PS, bias, STEP_J0 = 50_000, INTERPOLATION = True, VERBOSE = False):
    PS_2 = PS_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias)
    N2   = np.max(PS_2)
    PS_2 = PS_2 / N2
    Bessel = np.array([\
             np.array([special.j0(k*t*comoving_distance_z) for k in k_array])\
             for t in theta])
    A_sp = [splrep(k_array, hmf_PS * PS_2 * k_array / (2*np.pi) * Bessel[it], s=0, k=1) for it in range(len(theta))]
    R_T2 = np.array([splint(0, k_array[-1], A_sp[it]) for it in range(len(theta))])
    return R_T2 * N2, 0

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

def omega_z_component_singleCore_1halo(z, args):
    theta, M_DM_min, M_DM_max, \
    NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, \
    LOW_RES, STEP_J0, INTERPOLATION, VERBOSE = args
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
    oz1, e1 = omega_inner_integral_1halo(theta, z, comoving_distance_z, M_h_array, HMF_array,
                                        NCEN, NSAT, U_FT, k_array, hmf_PS, bias,
                                        STEP_J0, INTERPOLATION, VERBOSE)
    return oz1

def omega_z_component_singleCore_2halo(z, args):
    theta, M_DM_min, M_DM_max, \
    NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, \
    LOW_RES, STEP_J0, INTERPOLATION, VERBOSE = args
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
    return oz2

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

def omega_1halo_singleCore(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
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
    ### Single Core z integral ######################################################
    args = theta, M_DM_min, M_DM_max, \
            NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, \
            LOW_RES, STEP_J0, INTERPOLATION, VERBOSE
    itg = np.array([omega_z_component_singleCore_1halo(z, args) for z in z_array])
    #################################################################################
    #TODO: this calls init_lookuptable again, should distirbute it in (or as in) the omega_z_component_parallel
    N_G = get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
                         LOW_RES = LOW_RES, int_M_min=np.power(10, M_DM_min), int_M_max=np.power(10, M_DM_max))
    I1 = np.array([np.trapz(itg[:,i] * factor_z, z_array) for i in range(len(theta))])
    return I1/ np.power(N_G, 2)

def omega_2halo_singleCore(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
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
    ### Single Core z integral ######################################################
    args = theta, M_DM_min, M_DM_max, \
            NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, \
            LOW_RES, STEP_J0, INTERPOLATION, VERBOSE
    itg = np.array([omega_z_component_singleCore_2halo(z, args) for z in z_array])
    #################################################################################
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
    min_lnk, max_ln_k, step_lnk = -11.5, 13.6, 0.001
    if LOW_RES:
        FOLDERPATH = _HERE_PATH + '/HOD/HMF_tables/LowRes/'
        min_lnk, max_ln_k, step_lnk = -11.5, 16.6, 0.05
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
