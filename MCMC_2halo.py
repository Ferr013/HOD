#!/usr/bin/python3
################################################################################################################
# Giovanni Ferrami July 2024
################################################################################################################
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from scipy import signal
import emcee
import sys, os
sys.path.append(".")
import HOD
from halomod.bias import Tinker10
from astropy.cosmology import Planck15
cosmo = Planck15
sigma_8 = 0.8159
h = cosmo.H(0).value/100
c_light = 299792.458 #speed of light km/s

### HOD preloading the tables ############################################################################
def load_tables_and_cosmo(z, LOW_RES, M_DM_min, M_DM_max, PRECOMP_UFT = False, REWRITE_TBLS = False):
    if PRECOMP_UFT:
        M_h_array, HMF_array, nu_array, k_array, hmf_PS, U_FT =\
            HOD.init_lookup_table(z, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
    else:
        M_h_array, HMF_array, nu_array, k_array, hmf_PS = \
            HOD.init_lookup_table(z, PRECOMP_UFT, REWRITE_TBLS, LOW_RES, M_DM_min, M_DM_max)
        crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
        U_FT = np.array([HOD.u_FT(k, M_h_array, z, crit_dens_rescaled) for k in k_array])
    bias = Tinker10(nu=nu_array, sigma_8 = sigma_8, cosmo = cosmo).bias()
    comoving_distance_z = cosmo.comoving_distance(z).value
    return M_h_array, HMF_array, k_array, hmf_PS, U_FT, bias, comoving_distance_z

def omega_z_component_singleCore_2halo(z, args, tables_and_cosmo):
    theta, M_DM_min, M_DM_max, \
    NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, \
    LOW_RES, STEP_J0, INTERPOLATION, VERBOSE = args
    M_h_array, HMF_array, k_array, hmf_PS, \
    U_FT, bias, comoving_distance_z = tables_and_cosmo
    if VERBOSE: print('len (M_h_array) @ z = ',z,' : ', len(M_h_array))
    oz2, e2 = HOD.omega_inner_integral_2halo(theta, z, comoving_distance_z, M_h_array, HMF_array,
                                        NCEN, NSAT, U_FT, k_array, hmf_PS, bias,
                                        STEP_J0, INTERPOLATION, VERBOSE)
    return oz2

def omega_2halo_singleCore(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
                            _tables_and_cosmo_, M_DM_min, M_DM_max,
                            PRECOMP_UFT = False, REWRITE_TBLS = False,
                            LOW_RES = False, STEP_J0 = 50_000, cores=None,
                            INTERPOLATION = False, VERBOSE = False):
    if VERBOSE: print('M_DM_min, M_DM_max = ', M_DM_min, M_DM_max)
    M_h_array, _, __, ___, ____, _____, ______ = _tables_and_cosmo_[0]
    if VERBOSE: print('len (M_h_array) @ z = 0 : ', len(M_h_array))
    NCEN = HOD.N_cen(M_h_array, M_min, sigma_logM)
    NSAT = HOD.N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z = np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
####### Single Core z integral #################################################################################
    args = theta, M_DM_min, M_DM_max, \
            NCEN, NSAT, PRECOMP_UFT, REWRITE_TBLS, \
            LOW_RES, STEP_J0, INTERPOLATION, VERBOSE
    itg = np.array([omega_z_component_singleCore_2halo(z, args, _tables_and_cosmo_[i])\
                    for i, z in enumerate(z_array)])
################################################################################################################
    #TODO: this calls init_lookuptable again, should distirbute it in (or as in) the omega_z_component_parallel
    N_G = HOD.get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
                         LOW_RES = LOW_RES, int_M_min=np.power(10, M_DM_min), int_M_max=np.power(10, M_DM_max))
    I2 = np.array([np.trapz(itg[:,i] * factor_z, z_array) for i in range(len(theta))])
    return I2/ np.power(N_G, 2)

def get_Mh_interval(mag_min = 0, mag_max = np.inf):
    if mag_min == 0 or mag_max == np.inf:
        M_DM_min, M_DM_max = 0, np.inf
    else:
        M_DM_min, M_DM_max = HOD.get_M_DM_range(np.mean(z_array), mag_max, mag_min, delta_z=0.5)
    return M_DM_min, M_DM_max
################################################################################################################

def get_Nico_obs(z, MIN_THETA = 0):
    fnames = ['ACF_new_parameters_z5.5.txt',
            'ACF_new_parameters_z6.5.txt',
            'ACF_new_parameters_z7.4.txt',
            'ACF_new_parameters_z8.5.txt',
            'ACF_new_parameters_z9.3.txt',
            'ACF_new_parameters_z10.6.txt',
            'ACF_new_parameters_z11.5.txt']

    fname = fnames[int(int(z//1) - 5)]
    data = pd.read_csv('Data_Nico/'+fname, sep=' ')
    bin_centre = data['theta_bin'].to_numpy()
    w_obs = data['w_theta'].to_numpy()
    w_err = data['err_w_theta'].to_numpy()
    z_array = data['z_array'].to_numpy()
    Nz = data['Nz'].to_numpy()
    z_array, Nz = z_array[z_array>0], Nz[z_array>0]
    N_norm = Nz / (np.sum(Nz) * np.diff(z_array)[0])
    mask = bin_centre > MIN_THETA
    return bin_centre[mask], w_obs[mask], w_err[mask], z_array, N_norm

def log_likelihood(theta):
    logM_min, logM_sat = theta
    M_min, M_sat = np.power(10, logM_min), np.power(10, logM_sat)
    sigma_logM, alpha = 0.2, 1.0
    o2_model = omega_2halo_singleCore(bin_centre/206265,
                                        M_min, sigma_logM, M_sat, alpha,
                                        N_norm, z_array,
                                        _tables_and_cosmo_,
                                        M_DM_min, M_DM_max,
                                        LOW_RES=COARSE)
    chi2 = np.power((o2_model-w_obs)/w_err,2)
    return np.sum(chi2)

def log_prior(theta):
    logM_min, logM_sat = theta
    if  9 < logM_min < 13 and\
        9 < logM_sat < 13:
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

def init_walkers(PERC = 0.05):
    logM_min_g, logM_sat_g = 11, 12
    theta_init = np.array([logM_min_g, logM_sat_g])
    return theta_init + PERC * np.random.randn(8, 2)

################################################################################################################
MAX_ITER = 20
PROGRESS = True
RESET = True
COARSE = True
R_SEED = 753
### !!! Change parameters below for each run !!! ###############################################################
Z_OBSERVATION = 5.5
mag_min, mag_max = -22.3, -15.5
MIN_THETA = 5 #arcsec

argv = sys.argv
Z_OBSERVATION = argv[1]
mag_min, mag_max = argv[2], argv[3]
MIN_THETA = argv[4] #arcsec
################################################################################################################
filename = 'MCMC_chains/chains_2halo_z'+str(Z_OBSERVATION)+'.h5'
################################################################################################################

if PROGRESS: print('Computing MCMC over 2 halo term')

bin_centre, w_obs, w_err, z_array, N_norm = get_Nico_obs(Z_OBSERVATION, MIN_THETA=MIN_THETA)
if PROGRESS: print('Got Obs', w_obs.shape)

M_DM_min, M_DM_max = get_Mh_interval(mag_min, mag_max)
if PROGRESS: print(f'Got halo mass integration bounds: {M_DM_min:.2f} - {M_DM_max:.2f}')

_tables_and_cosmo_ = []
for z in z_array:
    _tables_and_cosmo_.append(load_tables_and_cosmo(z, COARSE, M_DM_min, M_DM_max))
    if PROGRESS: print(f'Loaded redshift {z} HMF/xPS tables and cosmological quantities')

init_pos = init_walkers()
nwalkers, ndim = init_pos.shape
backend = emcee.backends.HDFBackend(filename)
if RESET:
    backend.reset(nwalkers, ndim)
if PROGRESS and not RESET:
    if os.path.isfile(filename):
        print(f"Initial size: {backend.iteration}")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)
if not RESET:
    try:
        init_pos = sampler.get_last_sample()
    except:
        print('No backend found, starting from scratch')
# Track change in average autocorrelation time estimate and test convergence
_index, _autocorr, old_tau = 0, np.empty(MAX_ITER), np.inf
if PROGRESS: print('Start MCMC')
for sample in sampler.sample(init_pos,
                            iterations=MAX_ITER,
                            progress=PROGRESS):
    if sampler.iteration % 10:
        continue
    # Compute the autocorrelation time so far
    tau = sampler.get_autocorr_time(tol=0)
    _autocorr[_index] = np.mean(tau)
    _index += 1
    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
if RESET:
    np.savetxt(filename.split('.')[0]+'_autocorr.txt', _autocorr)
else:
    with open(filename.split('.')[0]+'_autocorr.txt', "a") as f: np.savetxt(f, _autocorr)
if PROGRESS:
    print(f"Final size: {backend.iteration}")
################################################################################################################
