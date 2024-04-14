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

cosmo = Planck15 #FlatLambdaCDM(H0=67.74, Om0=0.3089, Tcmb0=2.725)
sigma_8 = 0.8159
h = cosmo.H(0).value/100
c_light  = 299792.458 #speed of light km/s
###################################################################################################
### HALO OCCUPATION DISTRIBUTION ##################################################################
def N_cen(M_h, M_min, sigma_logM, DC = 1):
    return DC * 0.5*(1+special.erf((np.log10(M_h)-np.log10(M_min))/(np.sqrt(2)*sigma_logM)))

def N_sat(M_h, M_sat, alpha, M_min, sigma_logM, DC = 1):
    M_cut = np.power(M_min, -0.5) #Harikane 2018
    return DC * N_cen(M_h, M_min, sigma_logM) * np.power((M_h-M_cut)/M_sat,alpha)

def N_tot(M_h, M_sat, alpha, M_min, sigma_logM, DC = 1):
    return DC * (N_cen(M_h, M_min, sigma_logM) + N_sat(M_h, M_sat, alpha, M_min, sigma_logM))

def get_c_from_M_h(M_h, z):
    if 0:
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

def PS_1_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias):
    PS1cs = np.trapz(HMF_array * NCEN * NSAT * U_FT, M_h_array) * 2
    PS1ss = np.trapz(HMF_array * NSAT * NSAT * U_FT * U_FT, M_h_array) * 1
    PS_2h = np.power(np.trapz((NCEN + NSAT) * HMF_array * bias * U_FT, M_h_array), 2)
    return np.array([(PS1cs + PS1ss), PS_2h])

def integrate_between_J_zeros(theta, comoving_distance_z,
                              k_array, hmf_PS, PS_1, PS_2,
                              STEP_J0 = 5_000):
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
    for i in range(STEP_J0-1):
        mask_k = np.logical_and(k_array> k0[i], k_array< k0[i+1])
        k_intr = np.append(k0[i], np.append(k_array[mask_k], k0[i+1]))
        Bessel = np.array([special.j0(k*theta*comoving_distance_z) for k in k_array[mask_k]])
        int1 = simpson(np.pad(PS_1[mask_k], 1)
                        * k_intr / (2*np.pi) * np.pad(Bessel, 1), k_intr)
        int2 = simpson(np.pad(hmf_PS[mask_k], 1) * np.pad(PS_2[mask_k],1)
                        * k_intr / (2*np.pi) * np.pad(Bessel, 1), k_intr)
        res1 = np.append(res1, int1)
        res2 = np.append(res2, int2)
    return np.sum(res1), np.sum(res2)

def omega_inner_integral(theta, z, comoving_distance_z, M_h_array, HMF_array, NCEN, NSAT, U_FT,
                         k_array, hmf_PS, bias, STEP_J0 = 5_000):
    PS_1, PS_2 = PS_1_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias)
    N1, N2 = np.max(PS_1), np.max(PS_2)
    PS_1, PS_2 = PS_1 / N1, PS_2 / N2
    if 1:
        R_T = np.asarray([integrate_between_J_zeros(t, comoving_distance_z,
                              k_array, hmf_PS, PS_1, PS_2, STEP_J0) for t in theta])
        R_T1, R_T2 = R_T.T[0], R_T.T[1]
        return R_T1 * N1, R_T2 * N2
    Bessel = np.array([\
             np.array([special.j0(k*t*comoving_distance_z) for k in k_array])\
             for t in theta])
    R_T1 = np.trapz(PS_1 * k_array / (2*np.pi) * Bessel, k_array, axis = -1)
    R_T2 = np.trapz(hmf_PS * PS_2 * k_array / (2*np.pi) * Bessel, k_array, axis = -1)
    return R_T1 * N1, R_T2 * N2

def init(mem):
    global mem_id
    mem_id = mem
    return

def omega_z_component_single(args):
    job_id, shape, z, _args_ = args
    theta, NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, STEP_J0 = _args_
    shmem = SharedMemory(name=f'{mem_id}', create=False)
    shres = np.ndarray(shape, buffer=shmem.buf, dtype=np.float64)
    if PRECOMP_UFT:
        M_h_array, HMF_array, nu_array, k_array, hmf_PS, U_FT =\
            init_lookup_table(z, PRECOMP_UFT, REWRITE_TBLS, LOW_RES)
    else:
        M_h_array, HMF_array, nu_array, k_array, hmf_PS = \
            init_lookup_table(z, PRECOMP_UFT, REWRITE_TBLS, LOW_RES)
        crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
        U_FT = np.array([u_FT(k, M_h_array, z, crit_dens_rescaled) for k in k_array])
    bias = Tinker10(nu=nu_array, sigma_8 = sigma_8, cosmo = cosmo).bias()
    comoving_distance_z = cosmo.comoving_distance(z).value
    oz1, oz2 = omega_inner_integral(theta, z, comoving_distance_z, M_h_array, HMF_array,
                                    NCEN, NSAT, U_FT, k_array, hmf_PS, bias, STEP_J0)
    shres[job_id, 0, :], shres[job_id, 1, :] = oz1, oz2
    return

def omega_z_component_parallel(z_array, theta_array, NCEN, NSAT,
                               PRECOMP_UFT = False, REWRITE_TBLS = False,
                               LOW_RES = False, STEP_J0 = 5_000, cores=None):
    if cores is None:
        if len(z_array) <= multiprocessing.cpu_count():
            cores = len(z_array)
        else:
            print('MORE Z BINS THAN CORES, THE CODE IS NOT SMART ENOUGH TO HANDLE THIS YET')
            raise ValueError
            #cores = multiprocessing.cpu_count()
    _args_ = theta_array, NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, STEP_J0
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

def omega(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
          PRECOMP_UFT = False, REWRITE_TBLS = False,
          LOW_RES = True, STEP_J0 = 8_000, cores=None):
    if PRECOMP_UFT:
        M_h_array, __, ___, ____, _____, _______ = init_lookup_table(0, PRECOMP_UFT, REWRITE_TBLS, LOW_RES)
    else:
        M_h_array, __, ___, ____, _____ = init_lookup_table(0, PRECOMP_UFT, REWRITE_TBLS, LOW_RES)
    NCEN = N_cen(M_h_array, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z = np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    itg = omega_z_component_parallel(z_array, theta, NCEN, NSAT,
                                     PRECOMP_UFT, REWRITE_TBLS,
                                     LOW_RES, STEP_J0, cores)
    #TODO: this calls init_lookuptable again, should distirbute it in (or as in) the omega_z_component_parallel
    N_G = get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm, LOW_RES)
    I1 = np.array([np.trapz(itg[:,0,i] * factor_z, z_array) for i in range(len(theta))])
    I2 = np.array([np.trapz(itg[:,1,i] * factor_z, z_array) for i in range(len(theta))])
    return I1/ np.power(N_G, 2), I2/ np.power(N_G, 2)
###################################################################################################
#### INITIALIZE HMF ###############################################################################
def init_lookup_table(z, PRECOMP_UFT = False, REWRITE_TBLS = False, LOW_RES = False):
    _HERE_PATH = os.path.dirname(os.path.abspath(''))
    FOLDERPATH = _HERE_PATH + '/HOD/HMF_tables/'
    min_lnk, max_ln_k, step_lnk = -11.5, 12.6, 0.0001
    if LOW_RES:
        FOLDERPATH = _HERE_PATH + '/HOD/HMF_tables/LowRes/'
        min_lnk, max_ln_k, step_lnk = -11.5, 12.6, 0.0005
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
            hmf = MassFunction(Mmin = 9, Mmax = 18, dlog10m = 0.025,
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
        if PRECOMP_UFT:
            return hmf_mass, hmf_dndm, hmf_nu, hmf_k, hmf_PS, hmf_U_FT
        else:
            return hmf_mass, hmf_dndm, hmf_nu, hmf_k, hmf_PS
    else:
        print(FOLDERPATH)
        raise ValueError('Folder does not exist.')
###################################################################################################
### AVG QUANTITIES ################################################################################
def gal_density_n_g(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, DC = 1):
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM, DC)
    return np.trapz(HMF_array*NTOT, M_h_array)

def get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm, LOW_RES = False, DC = 1):
    _N_G, _dVdz = np.zeros(0),  np.zeros(0)
    for z in z_array:
        M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z, False, False, LOW_RES)
        _N_G  = np.append(_N_G, gal_density_n_g(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, DC))
        _dVdz = np.append(_dVdz, cosmo.comoving_distance(z).value**2 * c_light / cosmo.H(z).value)
    return np.trapz(_N_G * _dVdz * N_z_nrm, z_array)/np.trapz(_dVdz * N_z_nrm, z_array)

def get_AVG_N_tot(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, n_g=None):
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(HMF_array*NTOT, M_h_array)/n_g
    return np.trapz(HMF_array*NTOT, M_h_array)/np.trapz(HMF_array, M_h_array)

def get_AVG_Host_Halo_Mass(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, n_g=None):
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(M_h_array*HMF_array*NTOT, M_h_array)/n_g
    return np.trapz(M_h_array*HMF_array*NTOT, M_h_array)/np.trapz(HMF_array*NTOT, M_h_array)

def get_EFF_gal_bias(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, nu_array, n_g=None):
    bias = Tinker10(nu=nu_array, sigma_8 = sigma_8, cosmo = cosmo).bias()
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(bias*HMF_array*NTOT, M_h_array)/n_g
    return np.trapz(bias*HMF_array*NTOT, M_h_array)/np.trapz(HMF_array*NTOT, M_h_array)

def get_AVG_f_sat(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, n_g=None):
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(HMF_array*NSAT, M_h_array)/n_g
    return np.trapz(HMF_array*NSAT, M_h_array)/np.trapz(HMF_array*NTOT, M_h_array)
###################################################################################################
