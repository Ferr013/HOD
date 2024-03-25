import warnings
warnings.simplefilter("ignore")

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, special
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import uuid
sys.path.append(".")
import HOD
from halomod.bias import Tinker10

from tqdm import tqdm

from astropy.cosmology import FlatLambdaCDM
cosmo  = FlatLambdaCDM(H0=67.74, Om0=0.3089, Tcmb0=2.725)
OmegaM = cosmo.Om(0)
OmegaL = cosmo.Ode(0)
OmegaK = cosmo.Ok(0)
OmegaB = 0.049
OmegaC = OmegaM-OmegaB
H0 = cosmo.H(0).value
h  = H0/100
c_light  = 299792.458

############ Test integration ############
def func(x, theta, M_min, sigma_logM, M_sat, alpha, z, comoving_distance_z,
                 crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT, bias,
                 hmf_k, hmf_PS, D_ratio, _PS_NORM_):
    k = x / (theta / 206265 * comoving_distance_z)
    PS_1 = HOD.PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,
                     M_h_array, HMF_array, N_G, NCEN, NSAT)
    PS_2 = HOD.PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,
                     M_h_array, HMF_array, N_G, NCEN, NSAT, bias, hmf_k, hmf_PS,
                     D_ratio, _PS_NORM_, USE_MY_PS = 1)
    factor = HOD.factor_k(k, theta/206265, comoving_distance_z)
    return (PS_1 + PS_2) * factor

def init(mem):
    global mem_id
    mem_id = mem
    return

def make_integral_array(args):
    _args_, j0zs, shape, nzeros_node, job_id = args
    shmem = SharedMemory(name=f'{mem_id}', create=False)
    intgr = np.ndarray(shape, buffer=shmem.buf, dtype=np.float64)
    start = job_id * nzeros_node
    end = start + nzeros_node
    if 1:
        intgr[start:end] = [integrate.quad(func, j0z[0], j0z[1],
                            epsabs=1e-3, epsrel=1e-3, args = (_args_))[0]
                            for j0z in j0zs[start:end]]
    else:
        INTSTPS = 16
        theta, M_min, sigma_logM, M_sat, alpha, z, comoving_distance_z,\
        crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT, bias,\
        hmf_k, hmf_PS, D_ratio, _PS_NORM_ = _args_
        intgr[start:end] = [np.trapz(np.linspace(j0z[0], j0z[1], INTSTPS), [func(_,\
                            theta, M_min, sigma_logM, M_sat, alpha, z, comoving_distance_z,\
                            crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT, bias,\
                            hmf_k, hmf_PS, D_ratio, _PS_NORM_)\
                            for _ in np.linspace(j0z[0], j0z[1], INTSTPS)])\
                            for j0z in j0zs[start:end]]
    return

def make_integral_array_sharemem(_args_, nzeros, j0zs, cores=None):
    if cores is None:
        cores = multiprocessing.cpu_count()
    # print(f'# Cores: {cores}')
    args =  [(_args_, j0zs, nzeros, nzeros//int(cores), i) for i in range(int(cores))]
    exit = False
    try:
        global mem_id
        mem_id = str(uuid.uuid1())[:30] #Avoid OSError: [Errno 63]
        nbytes = nzeros * np.float64(1).nbytes
        shd_mem = SharedMemory(name=f'{mem_id}', create=True, size=nbytes)
        method = 'spawn'
        if sys.platform.startswith('linux'):
            method = 'fork'
        ctx = multiprocessing.get_context(method)
        pool = ctx.Pool(processes=cores, maxtasksperchild=1,
                        initializer=init, initargs=(mem_id,)
                        )
        try:
            pool.map_async(make_integral_array, args, chunksize=1).get(timeout=10_000)
        except KeyboardInterrupt:
            print("Caught kbd interrupt")
            pool.close()
            exit = True
        else:
            pool.close()
            pool.join()
            local_itgrd = np.ndarray(nzeros, buffer=shd_mem.buf, dtype=np.float64).copy()
    finally:
        shd_mem.close()
        shd_mem.unlink()
        if exit:
            sys.exit(1)
    return local_itgrd

def parallel_omega(theta, M_min, sigma_logM, M_sat, alpha, z, _PS_NORM_, NCORES=8, VERBOSE = False):
    MAX_REL_ERR = 1e-2
    pNZEROS, NZEROS, NZEROS_MAX = 0, 1_024, 32_768  #0, 2e10, 2e15
    J0_zeros = np.append(1e-3,special.jn_zeros(0, NZEROS_MAX))
    resint, relerr = np.zeros(1), 100
    NEXT_IT = True

    M_h_array, HMF_array, nu_array, hmf_k, hmf_PS = HOD.init_lookup_table(z, REWRITE_TBLS=False)
    NCEN = HOD.N_cen(M_h_array, M_min, sigma_logM)
    NSAT = HOD.N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    D_ratio   = (HOD.D_growth_factor(z)/HOD.D_growth_factor(0))**2 if z != 0 else 1
    bias = Tinker10(nu=nu_array).bias() if 1 else HOD.halo_bias_TINKER(nu_array)
    N_G  = HOD.n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array)
    comoving_distance_z = cosmo.comoving_distance(z).value
    crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)

    __args__ = theta, M_min, sigma_logM, M_sat, alpha, z, comoving_distance_z,\
    crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT, bias,\
    hmf_k, hmf_PS, D_ratio, _PS_NORM_

    while NEXT_IT:
        mem_id = None
        j0zs = np.append(J0_zeros[pNZEROS],
                         np.repeat(J0_zeros[pNZEROS+1:NZEROS+1],2)[:-1]).reshape(-1,2)
        curr_itgA = make_integral_array_sharemem(__args__, NZEROS - pNZEROS, j0zs, cores=NCORES)
        curr_itgS = np.sum(curr_itgA)
        resint = np.append(resint, resint[-1] + curr_itgS)
        if VERBOSE:
            print('##################')
            print('CALC INTEGRAL')
            print('zeros')
            print(j0zs[0], j0zs[1], j0zs[-1])
            print('SHAPE MEM: ', curr_itgA.shape)
            print('integral')
            print(curr_itgS)
            print('##################')
            k_min = np.log10(np.min(J0_zeros[pNZEROS+1:NZEROS]/\
            (theta/206265*comoving_distance_z)))
            k_max = np.log10(np.max(J0_zeros[pNZEROS+1:NZEROS]/\
            (theta/206265*comoving_distance_z)))
            print(f'IT #: {len(resint) - 1} NZEROS : {NZEROS} -\
            (k_min: {k_min:.2f}, k_max: {k_max:.2f})')
            print(resint, curr_itgS)
            print(f'Rel. error wrt last it: {relerr:.1e}') \
                if len(resint) > 1 else print(f'Rel. error {999}')
        if len(resint) > 1: relerr = np.abs((resint[-1]-resint[-2])/resint[-1])
        pNZEROS, NZEROS = NZEROS, NZEROS * 2
        if relerr < MAX_REL_ERR or NZEROS > NZEROS_MAX: NEXT_IT = False
    return resint[-1], relerr

if __name__ == "__main__":
    z = 1.7
    M_sat, M_min, sigma_logM, alpha = 10**14.18, 10**12.46, 0.2, 1.0
    z_array = np.array([1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1])
    N_z_nrm = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z =  np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))

    _PS_NORM_ = HOD.norm_power_spectrum()

    result, error = np.zeros(0), np.zeros(0)
    _theta_arcsec = np.logspace(-0.5, 3.5, 7)
    _theta = _theta_arcsec * 1/206265 # 1 arcsec in rad
    _theta_arcsec = [1]
    for theta in tqdm(_theta_arcsec):
        r = [parallel_omega(theta, M_min, sigma_logM, M_sat, alpha, z, _PS_NORM_,\
                            NCORES = 8, VERBOSE = False)[0] for z in tqdm(z_array)]
        print(r)
        result = np.append(result, np.trapz(np.array(r) * factor_z, z_array))
    # print(f'RESULT: {result[0]* _factor_z:.2e} | err: {result[1]:.2e}')
    print(result, factor_z)
