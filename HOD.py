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
cosmo = Planck15 #FlatLambdaCDM(H0=67.74, Om0=0.3089, Tcmb0=2.725)
h = cosmo.H(0).value/100
c_light  = 299792.458 #speed of light km/s
###################################################################################################
### HALO OCCUPATION DISTRIBUTION ##################################################################
def N_cen(M_h, M_min, sigma_logM):
    return 0.5*(1+special.erf((np.log10(M_h)-np.log10(M_min))/(np.sqrt(2)*sigma_logM)))

def N_sat(M_h, M_sat, alpha, M_min, sigma_logM):
    M_cut = np.power(M_min, -0.5) #Harikane 2018
    return N_cen(M_h, M_min, sigma_logM)*np.power((M_h-M_cut)/M_sat,alpha)

def N_tot(M_h, M_sat, alpha, M_min, sigma_logM, DC = 1):
    return DC * (N_cen(M_h, M_min, sigma_logM) + N_sat(M_h, M_sat, alpha, M_min, sigma_logM))

def n_g(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array):
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    return np.trapz(HMF_array*NTOT, M_h_array)

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

def PS_1_2h(M_h_array, HMF_array, N_G, NCEN, NSAT, U_FT, bias):
    PS1cs = np.trapz(HMF_array * NCEN * NSAT * U_FT, M_h_array) * 2 / np.power(N_G, 2)
    PS1ss = np.trapz(HMF_array * NSAT * NSAT * U_FT * U_FT, M_h_array) * 1 / np.power(N_G, 2)
    PS_2h = np.power(np.trapz((NCEN + NSAT) * HMF_array * bias * U_FT, M_h_array)/ N_G, 2)
    return np.array([PS1cs + PS1ss, PS_2h])

def omega_inner_integral(theta, z, comoving_distance_z, M_h_array, HMF_array, N_G, NCEN, NSAT, U_FT,
                         k_array, hmf_PS, bias):
    PS_1, PS_2 = PS_1_2h(M_h_array, HMF_array, N_G, NCEN, NSAT, U_FT, bias)
    Bessel = np.array([\
             np.array([special.j0(k*t*comoving_distance_z) for k in k_array])\
             for t in theta])
    _com_phys_ = 1 / (1 + z)
    R_T1 = np.trapz(PS_1 * k_array / (2*np.pi) * Bessel * _com_phys_, k_array, axis = -1)
    R_T2 = np.trapz(hmf_PS * PS_2 * k_array / (2*np.pi) * Bessel * _com_phys_, k_array, axis = -1)
    return R_T1, R_T2

def init(mem):
    global mem_id
    mem_id = mem
    return

def omega_z_component_single(args):
    job_id, shape, z, _args_ = args
    theta, NCEN, NSAT, REWRITE_TBLS = _args_
    shmem = SharedMemory(name=f'{mem_id}', create=False)
    shres = np.ndarray(shape, buffer=shmem.buf, dtype=np.float64)
    M_h_array, HMF_array, nu_array, k_array, hmf_PS, U_FT = init_lookup_table(z, REWRITE_TBLS)
    bias = Tinker10(nu=nu_array).bias()
    N_G = np.trapz(HMF_array*(NCEN + NSAT), M_h_array)
    comoving_distance_z = cosmo.comoving_distance(z).value
    oz1, oz2 = omega_inner_integral(theta, z, comoving_distance_z, M_h_array, HMF_array,
                                    N_G, NCEN, NSAT, U_FT, k_array, hmf_PS, bias)
    shres[job_id, 0, :], shres[job_id, 1, :] = oz1, oz2
    return

def omega_z_component_parallel(z_array, theta_array, NCEN, NSAT, REWRITE_TBLS, cores=None):
    if cores is None:
        if len(z_array) <= multiprocessing.cpu_count():
            cores = len(z_array)
        else:
            print('MORE Z BINS THAN CORES, THE CODE IS NOT SMART ENOUGH TO HANDLE THIS YET')
            raise ValueError
            #cores = multiprocessing.cpu_count()
    _args_ = theta_array, NCEN, NSAT, REWRITE_TBLS
    shape = (int(cores), 2, len(theta_array))
    args =  [(i, shape, z_array[i], _args_) for i in range(int(cores))]
    exit = False
    try:
        global mem_id
        mem_id = str(uuid.uuid1())[:30] #Avoid OSError: [Errno 63]
        nbytes = (2 + int(cores) + len(theta_array)) * np.float64(1).nbytes
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
          REWRITE_TBLS = False, cores=None):
    M_h_array, __, ___, ____, _____, _______ = init_lookup_table(0, REWRITE_TBLS)
    NCEN = N_cen(M_h_array, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z = np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    itg = omega_z_component_parallel(z_array, theta, NCEN, NSAT, REWRITE_TBLS, cores)
    I1 = np.array([np.trapz(itg[:,0,i] * factor_z, z_array) for i in range(len(theta))])
    I2 = np.array([np.trapz(itg[:,1,i] * factor_z, z_array) for i in range(len(theta))])
    return I1, I2
###################################################################################################
#### INITIALIZE HMF ###############################################################################
def init_lookup_table(z, REWRITE_TBLS = False):
    _HERE_PATH = os.path.dirname(os.path.abspath(''))
    FOLDERPATH = _HERE_PATH + '/HOD/HMF_tables/'
    if os.path.exists(FOLDERPATH):
        FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'.txt'
        if (os.path.isfile(FPATH) and not REWRITE_TBLS):
            hmf_mass, hmf_dndm, hmf_nu = np.loadtxt(FPATH, delimiter=',')
            FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'_PS.txt'
            hmf_k, hmf_PS = np.loadtxt(FPATH, delimiter=',')
            FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'_U.txt'
            hmf_U_FT = np.loadtxt(FPATH, delimiter=',')
        else:
            print(f'Calculating HMF table at redshift {z:.2f}')
            hmf = MassFunction(Mmin = 9, Mmax = 18, dlog10m = 0.025,
                               lnk_min = -11.1, lnk_max = 13.6, dlnk=0.00005,
                               z=z, hmf_model = "Behroozi", sigma_8 = 0.8159, cosmo_model = cosmo)
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
            crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
            hmf_U_FT = np.array([u_FT(k, hmf_mass, z, crit_dens_rescaled) for k in hmf_k])
            np.savetxt(FPATH, hmf_U_FT,  delimiter=',')
        return hmf_mass, hmf_dndm, hmf_nu, hmf_k, hmf_PS, hmf_U_FT
    else:
        print(FOLDERPATH)
        raise ValueError('Folder does not exist.')
###################################################################################################
### AVG QUANTITIES ################################################################################
def get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm):
    _N_G, _dVdz = np.zeros(0),  np.zeros(0)
    for z in z_array:
        M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
        _N_G  = np.append(_N_G, n_g(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array))
        _dVdz = np.append(_dVdz, cosmo.comoving_distance(z).value**2 * c_light / cosmo.H(z).value)
    return np.trapz(_N_G * _dVdz * N_z_nrm, z_array)/np.trapz(_dVdz * N_z_nrm, z_array)

def get_AVG_N_tot(M_min, sigma_logM, M_sat, alpha, z):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    return np.trapz(HMF_array*NTOT, M_h_array)/np.trapz(HMF_array, M_h_array)

def get_AVG_Host_Halo_Mass(M_min, sigma_logM, M_sat, alpha, z):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    return np.trapz(M_h_array*HMF_array*NTOT, M_h_array)/np.trapz(HMF_array*NTOT, M_h_array)

def get_EFF_gal_bias(M_min, sigma_logM, M_sat, alpha, z):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
    bias = Tinker10(nu=nu_array).bias()
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    return np.trapz(bias*HMF_array*NTOT, M_h_array)/np.trapz(HMF_array*NTOT, M_h_array)

def get_AVG_f_sat(M_min, sigma_logM, M_sat, alpha, z):
    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = init_lookup_table(z)
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    return np.trapz(HMF_array*NSAT, M_h_array)/np.trapz(HMF_array*NTOT, M_h_array)
###################################################################################################
