#!/usr/bin/python3

import numpy as np
import pandas as pd
from scipy import signal
import emcee
import sys, os
sys.path.append(".")
import HOD

def get_Nico_obs(z):
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
    return bin_centre, w_obs, w_err, z_array, N_norm

def log_likelihood(theta):
    M_min, M_sat = theta
    sigma_logM, alpha = 0.2, 1.0
    o2_model = HOD.omega_2halo(bin_centre, M_min, sigma_logM, M_sat, alpha,
                               N_norm, z_array, LOW_RES=COARSE, mag_min = -22.3, mag_max = -15.5)
    chi2 = np.power((o2_model-w_obs)/w_err,2)
    return np.sum(chi2)

def log_prior(theta):
    M_min, M_sat = theta
    if  1e9 < M_min < 1e13 and\
        1e9 < M_sat < 1e13:
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

def init_walkers(PERC = 0.05):
    M_min_g, M_sat_g = 1e11, 1e12
    theta_init = np.array([M_min_g, M_sat_g])
    return theta_init + PERC * np.random.randn(8, 2)

###################################################################
MAX_ITER = 20
PROGRESS = True
RESET = True
COARSE = True
R_SEED = 753
Z_OBSERVATION = 5.5
###################################################################
filename = 'chains_2halo_z'+str(Z_OBSERVATION)+'.h5'
###################################################################

if PROGRESS: print('Computing MCMC over 2 halo term')
bin_centre, w_obs, w_err, z_array, N_norm = get_Nico_obs(Z_OBSERVATION)

if PROGRESS: print('Got Obs', w_obs.shape)
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
