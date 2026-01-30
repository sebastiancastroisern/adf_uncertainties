 # Imports
import time
import os
import h5py
import emcee 
import corner
import argparse
import numpy             as np
import matplotlib.pyplot as plt
import wavefronts.params_config as pr
from typing          import *
from numba           import njit
from tqdm            import tqdm
from numpy.linalg    import cond
from multiprocessing import Pool, cpu_count
from emcee.moves     import DEMove, DESnookerMove, KDEMove, StretchMove
from wavefronts.wavefronts_SEB  import *

# Patch temporaire pour la compatibilité NumPy 2.0
if not hasattr(np, 'infty'):
    np.infty = np.inf

# ---------- Argument Parser ----------
parser = argparse.ArgumentParser(description='MCMC Reconstruction for Antenna Array Data')
parser.add_argument('--filepath', type=str, default='./test_NJ/', help='Path to the data files') # other example './addednoise_110uV_5antennas/'
parser.add_argument('--nmax', type=int, default=np.inf, help='Maximum number of coincidences to process')
parser.add_argument('--niter', type=int, default=3000 , help='Number of MCMC iterations per coincidence')
parser.add_argument('--nburnin', type=int, default=1000 , help='Number of burn-in iterations to discard')
parser.add_argument('--test', action='store_true', help='Flag to run a test MCMC on a single coincidence')
parser.add_argument('--multi', action='store_true', help='Flag to enable multiprocessing for MCMC processing')
parser.add_argument('--swf', action='store_true', help='Flag to run MCMC only on SWF parameters')
parser.add_argument('--startidx', type=int, default=0, help='Starting coincidence index for processing')
args = parser.parse_args()

# ---------- Parameters ----------

file_path = args.filepath                         # Path to data files
n_max     = args.nmax  if not args.test else 2    # Max number of coincidences to process
n_iter_v  = args.niter if not args.test else 100 # Number of MCMC iterations per coincidence
n_burnin_v  = args.nburnin if not args.test else 100  # Number of burn-in iterations to discard
start_idx_a = args.startidx                    # Starting coincidence index

# ---------- Parameters from Config ----------

bounds = pr.bounds

# ---------- EAS Params ----------

def restructure_data(file_path: str) -> None:
    """Convertir les dictionnaires en arrays indexables"""
    SWF_res = np.load(os.path.join(file_path, "SWF_res.npy"), allow_pickle=True).item()['data']
    ADF_res = np.load(os.path.join(file_path, "ADF_res.npy"), allow_pickle=True).item()['data']
    
    # Convertir en arrays 2D réguliers
    n_coinc = len(SWF_res)
    swf_array = np.array([SWF_res[i] for i in range(n_coinc)])  # shape: (n_coinc, 4)
    adf_array = np.array([ADF_res[i] for i in range(n_coinc)])  # shape: (n_coinc, 4)
    
    np.save(os.path.join(file_path, "SWF_array.npy"), swf_array)
    np.save(os.path.join(file_path, "ADF_array.npy"), adf_array)

    del SWF_res
    del ADF_res
    del swf_array
    del adf_array

@njit
def build_Xmax(swf_data: np.ndarray) -> np.ndarray:
    """ Build shower parameters for a given coincidence index
    Inputs:
        swf_data: SWF parameters [alpha_SWF (deg), beta_SWF (deg), rxmax_SWF (m), t_0 (s)]
        adf_data: ADF parameters [theta_ADF (deg), phi_ADF (deg), dw_ADF (m), Amp_ADF]
    Outputs:
        [Xmax (3D position in m), theta_ADF (rad), phi_ADF (rad), dw_ADF (m), Amp_ADF]"""


    alpha_SWF_rad = swf_data[0]
    beta_SWF_rad  = swf_data[1]
    rxmax_SWF     = swf_data[2]

    ca, sa, cb, sb = np.cos(alpha_SWF_rad), np.sin(alpha_SWF_rad), np.cos(beta_SWF_rad), np.sin(beta_SWF_rad)
    Xmax = np.array([rxmax_SWF * sa * cb, rxmax_SWF * sa * sb, rxmax_SWF * ca + pr.groundAltitude], dtype=np.float64)
    
    return Xmax

def load_shower_data(file_path: str, coinc_idx: int) -> Tuple[np.ndarray, Tuple]:
    """Load using memmap for efficient indexing"""
    if not os.path.exists(os.path.join(file_path, "SWF_array.npy")) or not os.path.exists(os.path.join(file_path, "ADF_array.npy")):
        restructure_data(file_path)

    # Memmap permet d'accéder à un index sans charger tout le fichier
    swf_data = np.load(os.path.join(file_path, "SWF_array.npy"), 
                       mmap_mode='r')[coinc_idx].copy()
    adf_data = np.load(os.path.join(file_path, "ADF_array.npy"),
                       mmap_mode='r')[coinc_idx].copy()

    swf_adf_params = swf_data.tolist() + adf_data.tolist()
    swf_adf_params[0] *= np.pi / 180  # alpha_SWF to rad
    swf_adf_params[1] *= np.pi / 180  # beta_SWF to rad
    swf_adf_params[4] *= np.pi / 180  # theta_ADF to rad
    swf_adf_params[5] *= np.pi / 180  # phi_ADF to rad
    
    nants = np.load(os.path.join(file_path, "co_nants.npy"), 
                    mmap_mode='r')[coinc_idx]
    peak_amp = np.load(os.path.join(file_path, "co_peak_amp_array.npy"),
                       mmap_mode='r')[coinc_idx, :nants].copy()
    ant_coords = np.load(os.path.join(file_path, "co_antenna_coords_array.npy"),
                         mmap_mode='r')[coinc_idx, :nants].copy()
    tants = np.load(os.path.join(file_path, "co_peak_time_array.npy"),
                    mmap_mode='r')[coinc_idx, :nants].copy()
    
    params = swf_adf_params
    data = (peak_amp, tants, ant_coords, params) # add params to transfer adf values directly
    return params, data


# ------------ SWF EMCEE Functions ------------

def log_prior_swf(params: np.ndarray, bounds: np.ndarray) -> float:
    """ Log prior function for MCMC """
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    lprior = 0.0
    whithin_bounds = np.all((params >= lower_bounds) & (params <= upper_bounds))
    if not whithin_bounds:
        # if not bool_adf:
        #     if params[2] > upper_bounds[2] or params[2] < lower_bounds[2]:
        #         print("rxmax out of bounds")
        #         print("params[2] =", params[2])
        #     if params[3] > upper_bounds[3] or params[3] < lower_bounds[3]:
        #         print("t0 out of bounds")
        #         print("params[3] =", params[3])
        #     if params[6] > upper_bounds[6] or params[6] < lower_bounds[6]:
        #         print("dw out of bounds")
        #         print("params[6] =", params[6])
        #     if params[7] > upper_bounds[7] or params[7] < lower_bounds[7]:
        #         print("Amp out of bounds")
        #         print("params[7] =", params[7])
        #     if params[4] > upper_bounds[4] or params[4] < lower_bounds[4]:
        #         print("theta out of bounds")
        #         print("params[4] =", params[4])
        #     if params[5] > upper_bounds[5] or params[5] < lower_bounds[5]:
        #         print("phi out of bounds")
        #         print("params[5] =", params[5])
        #     if params[0] > upper_bounds[0] or params[0] < lower_bounds[0]:
        #         print("alpha out of bounds")
        #         print("params[0] =", params[0])
        #     if params[1] > upper_bounds[1] or params[1] < lower_bounds[1]:
        #         print("beta out of bounds")
        #         print("params[1] =", params[1])
        return np.inf
    return lprior

def log_likelihood_swf(params: np.ndarray, data: Tuple) -> float:
    """ Log likelihood function for MCMC """
    _, tants, ant_coords, _ = data
    swf_params = params[:4]

    swf_params = (swf_params[0], swf_params[1], swf_params[2], swf_params[3])
    chi2_t = SWF_loss(swf_params, ant_coords, tants, False) # if true returns chi2/ndof

    return -0.5 * chi2_t

def log_posterior_swf(params: np.ndarray, data: Tuple, bounds: np.ndarray) -> float:
    lp = log_prior_swf(params, bounds)
    return -np.inf if not np.isfinite(lp) else lp + log_likelihood_swf(params, data)

def mcmc_single_idx_swf(params: np.ndarray, data: Tuple, n_steps:int = 1000, n_burnin:int = 500, n_walkers:int = 16, progress:bool = True, thinning:bool = True, bounds :np.ndarray=pr.bounds) -> Tuple[np.ndarray, np.ndarray]:
    """Run MCMC for 1 coincidence converging on the first 4 parameters (SWF) only."""
    ndim_swf = 4
    initial_pos_swf = params[:ndim_swf] + 1e-4 * np.random.randn(n_walkers, ndim_swf)
    sampler_SWF = emcee.EnsembleSampler(n_walkers, ndim_swf, log_posterior_swf, args=[data, np.array(bounds[:ndim_swf])], moves=[DEMove(), DESnookerMove(), KDEMove()])

    # Bunrn-in
    pos_swf = sampler_SWF.run_mcmc(initial_pos_swf, n_burnin, progress=progress)
    sampler_SWF.reset()

    # Main sampling
    pos_swf = sampler_SWF.run_mcmc(pos_swf, n_steps, progress=progress)
    samples_swf = sampler_SWF.get_chain(flat=True)
    log_prob_swf = sampler_SWF.get_log_prob(flat=True)
    del sampler_SWF

    return samples_swf, log_prob_swf if not thinning else log_prob_swf[::10]

def mcmc_sequential_swf(file_path: str, n_max: int, n_iter: int, n_burnin: int, n_walkers: int=16, progress: bool=True, thinning: bool=False) -> None:
    """ Run MCMC EMCEE for all coincidences up to n_max  seqeuntially"""
    new_path     = os.path.join(file_path, "corner_plots_emcee_swf/") ; os.makedirs(new_path, exist_ok=True)
    nants        = np.load(os.path.join(file_path, "co_nants.npy"))
    n_coinc      = len(nants)
    n_to_process = min(n_coinc, n_max)

    for coinc_idx in range(n_to_process):
        params, data = load_shower_data(file_path, coinc_idx)

        if progress:
            print(f"\nRunning MCMC EMCEE for coincidence {coinc_idx+1}/{n_to_process} ({nants[coinc_idx]} antennas)...")

        samples, log_prob = mcmc_single_idx_swf(params, data,
                                                 n_steps=n_iter,
                                                 n_burnin=n_burnin,
                                                 n_walkers=n_walkers,
                                                 progress=progress,
                                                 thinning=thinning)
      
        if progress:
            print(f"\nMCMC EMCEE completed for coincidence {coinc_idx+1}/{n_to_process}")

        # --- Sauvegarde incrémentale ---
        corner_plot_swf(samples, log_prob, save_path=new_path, coinc_idx=coinc_idx, file_path=file_path)
        save_samples_res(new_path, coinc_idx, samples, log_prob,
                          n_total=n_to_process)
    
    print(f"\n\n[Processed {n_to_process} coinc in MCMC EMCEE]")

def corner_plot_swf(samples: np.ndarray, log_prob: np.ndarray, save_path: str, coinc_idx: int, file_path: str) -> None:
    """ Corner plot of the first 4 paramters only (SWF)"""
    # Load the true values of the simulations
    SWF_res = np.load(os.path.join(file_path, 'SWF_res.npy'), allow_pickle=True).item()
    SWF_res = SWF_res['data']
    
    SWF_values = SWF_res[coinc_idx]
    SWF_values[0] *= np.pi / 180  # alpha to rad
    SWF_values[1] *= np.pi / 180  # beta to rad
    
    figure = corner.corner(
        samples[:, :4], 
        labels=["Alpha (def)", "Beta (deg)", "Rxmax (m)", "t0 (s)"],
        show_titles=True, 
        title_fmt=".2f", 
        quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 12},
        color='#2980b9',                # Bleu moderne
        hist_kwargs={
            'color': '#2980b9',
            'edgecolor': '#1f618d',
            'linewidth': 1.5,
            'alpha': 0.7
        },
        contour_kwargs={
            'colors': ['#2980b9', '#1f618d', '#154360'],  # Dégradé de bleu
            'linewidths': [1.0, 1.5, 2.0]
        },
        scatter_kwargs={
            'alpha': 0.2,
            's': 2,
            'color': '#2980b9',
            'rasterized': True
        },
        smooth=1.0,
        fill_contours=True,
        levels=(0.68, 0.95),
        truths=SWF_values,
        truth_color='#e74c3c',  # Rouge moderne

    )

    figure.savefig(
        os.path.join(save_path, f"corner_SWF_coinc_{coinc_idx:05d}.png"),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close(figure)
    print(f"Corner plot SWF saved for coincidence {coinc_idx} at {save_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(-log_prob, color='#2980b9')  # Bleu moderne
    plt.title(f'Log-Probability Trace for Coincidence {coinc_idx}', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Log-Probability', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(
        os.path.join(save_path, f"log_prob_trace_coinc_{coinc_idx:05d}.png"),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()
    print(f"Log-Probability trace saved for coincidence {coinc_idx} at {save_path}")

# ---------- Total MCMC Functions ----------
# function from the CC files

def log_prior(params: np.ndarray, bounds: np.ndarray) -> float:
    """ Log prior function for MCMC, physical assumptions on the probabilities """
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    whithin_bounds = np.all((params >= lower_bounds) & (params <= upper_bounds))
    if not whithin_bounds:
        # print('----------------------------------------------------')
        # if params[2] > upper_bounds[2] or params[2] < lower_bounds[2]:
        #     print("rxmax out of bounds")
        #     print("params[2] =", params[2])
        # if params[3] > upper_bounds[3] or params[3] < lower_bounds[3]:
        #     print("t0 out of bounds")
        #     print("params[3] =", params[3])
        # if params[6] > upper_bounds[6] or params[6] < lower_bounds[6]:
        #     print("dw out of bounds")
        #     print("params[6] =", params[6])
        # if params[7] > upper_bounds[7] or params[7] < lower_bounds[7]:
        #     print("Amp out of bounds")
        #     print("params[7] =", params[7])
        # if params[4] > upper_bounds[4] or params[4] < lower_bounds[4]:
        #     print("theta out of bounds")
        #     print("params[4] =", params[4])
        # if params[5] > upper_bounds[5] or params[5] < lower_bounds[5]:
        #     print("phi out of bounds")
        #     print("params[5] =", params[5])
        # if params[0] > upper_bounds[0] or params[0] < lower_bounds[0]:
        #     print("alpha out of bounds")
        #     print("params[0] =", params[0])
        # if params[1] > upper_bounds[1] or params[1] < lower_bounds[1]:
        #     print("beta out of bounds")
        #     print("params[1] =", params[1])
        return np.inf
    return 0.0

def log_likelihood(params: np.ndarray, data: Tuple) -> float:
    """ Log likelihood function for MCMC """
    peak_amp, tants, ant_coords, _ = data
    swf_params = params[:4]
    adf_params = params[4:]

    Xmax = build_Xmax(swf_params)
    
    chi2_a = ADF_loss(adf_params, peak_amp, ant_coords, Xmax, False) # if true returns chi2/ndof
    chi2_t = SWF_loss(swf_params, ant_coords, tants, False) # if true returns chi2/ndof

    return -0.5 * (chi2_t + chi2_a)

def log_post(params: np.ndarray, data: Tuple, bounds: np.ndarray) -> float:
    """ Log posterior function for MCMC, combining prior and likelihood """
    params_2pi = params.copy()
    # params_2pi[0] = params[0] % (np.pi)  # wrap alpha between 0 and 2pi
    # params_2pi[1] = params[1] % (2.0 * np.pi)  # wrap beta between 0 and 2pi
    # params_2pi[4] = params[4] % (np.pi)  # wrap theta between 0 and 2pi
    # params_2pi[5] = params[5] % (2.0 * np.pi)  # wrap phi between 0 and 2pi
    
    lp = log_prior(params_2pi, bounds)

    if np.isfinite(lp):
        return lp + log_likelihood(params_2pi, data)
    else:
        return -np.inf

def mcmc_single_idx(params: np.ndarray, data: Tuple, n_steps=1000, n_burnin=500, n_walkers=32, progress=True, thinning=True, bounds: np.ndarray=pr.bounds, return_mode: bool=False, thinning_factor: int=10) -> Tuple[np.ndarray, np.ndarray]:
    """ Run MCMC for 1 coincidence converging on all 8 parameters (SWF + ADF), with a burn-in phase.
     8 parameters: [alpha_SWF, beta_SWF, rxmax_SWF, t0, theta_ADF, phi_ADF, dw_ADF, Amp_ADF]
    Inputs : 
        params: Initial parameters from SWF + ADF fit
        data: Tuple containing (peak_amp, tants, ant_coords, params) for the loss calculations
        n_steps: Number of MCMC steps after burn-in
        n_burnin: Number of burn-in steps (between 10% and 30% of total steps)
        n_walkers: Number of walkers in the ensemble (should be > 2 * ndim)
        progress: Whether to show progress bar and prints
        thinning: Whether to thin the samples by a factor to reduce autocorrelation and file size
        bounds: Bounds for each parameter, already defined globally
        return_mode: Whether to return the modes of the distributions instead of all samples
        thinning_factor: Factor by which to thin the samples if thinning is True
    Outputs: 
        Tuple of samples and log probabilities or modes depending on return_mode value
        """
    # Initialize the sampler
    n_walkers = max(n_walkers, 16)  # Ensure a minimum number of walkers
    ndim = 8
    initial_pos = params +  np.random.randn(n_walkers, ndim) * 1e-2
    sampler_burnin = emcee.EnsembleSampler(n_walkers, ndim, log_post, args=[data, bounds], moves=[DEMove(), DESnookerMove(), StretchMove()])

    # Burn-in
    pos = sampler_burnin.run_mcmc(initial_pos, n_burnin, progress=progress, skip_initial_state_check=True)
    if progress:
        print(f"Condition number is {cond(pos.coords):.3e}")

    # Main sampling
    sampler  = emcee.EnsembleSampler(n_walkers, ndim, log_post, args=[data, bounds], moves=[DEMove(), DESnookerMove(), StretchMove()])
    pos      = sampler.run_mcmc(pos, n_steps, progress=progress, skip_initial_state_check=True)
    samples  = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)

    samples[:, 0] = samples[:, 0] % (1.0 * np.pi)  # wrap alpha between 0 and pi
    samples[:, 1] = samples[:, 1] % (2.0 * np.pi)  # wrap beta between 0 and 2pi
    samples[:, 4] = samples[:, 4] % (1.0 * np.pi)  # wrap theta between 0 and pi
    samples[:, 5] = samples[:, 5] % (2.0 * np.pi)  # wrap phi between 0 and 2pi
    samples[:, :2] = samples[:, :2] * 180 / np.pi  # convert alpha and beta to degrees
    samples[:,4:6] = samples[:,4:6] * 180 / np.pi  # convert theta and phi to degrees

    del sampler_burnin
    del sampler

    if return_mode: 
        modes = modes_finder(samples)
        return modes
    else: 
        if thinning: 
            samples, log_prob = samples[::thinning_factor], log_prob[::thinning_factor]
        return samples, log_prob
    
# ============================
#       BROKEN FUNCTION
# ============================
def mcmc_sequential(file_path: str, n_max: int, n_iter: int, n_burnin: int, n_walkers: int=32, progress: bool=True, thinning: bool=False, start_idx: int=0, corner_plot_bool: bool=False) -> None:
    """ Run MCMC EMCEE for all coincidences up to n_max seqeuntially"""
    # Initialise save directory and basic parameters
    nants        = np.load(os.path.join(file_path, "co_nants.npy"))
    n_coinc      = len(nants)
    n_to_process = min(n_coinc - start_idx, n_max)
    save_path    = os.path.join(file_path, "corner_plots_emcee/") ; os.makedirs(save_path, exist_ok=True)

    if n_to_process <= 0: # Edge case: no coincidences to process
        print("No coincidences to process from the given start index.")
        return

    # Loop over coincidences
    for coinc_idx in range(start_idx, start_idx+n_to_process):
        # load data first for current cioncidence
        params, data = load_shower_data(file_path, coinc_idx) 

        if progress:
            print(f"\nRunning MCMC EMCEE for coincidence {coinc_idx-start_idx+1}/{n_to_process} ({nants[coinc_idx]} antennas, {n_walkers} walkers, {n_iter} iterations, {n_burnin} burn-in, idx = {coinc_idx})...")

        # --- MCMC EMCEE ---
        samples, log_prob = mcmc_single_idx(params, data,
                                                n_steps=n_iter,
                                                n_burnin=n_burnin,
                                                n_walkers=n_walkers,
                                                progress=progress,
                                                thinning=thinning,
                                                bounds=pr.bounds,
                                                return_mode=False) # no mode return here, we save samples
        
        # --- Incremental save ---
        save_samples_res(file_path, coinc_idx, samples, log_prob,
                        n_total=n_to_process)
        if corner_plot_bool:
            corner_plot(samples, save_path=save_path, coinc_idx=coinc_idx)

        if progress:
                print(f"\nMCMC EMCEE completed for coincidence {coinc_idx-start_idx+1}/{n_to_process}, coincidence {coinc_idx}.")
            
    # --- End of all coincidences ---
    print(f"\n\n[Processed {n_to_process} coinc in MCMC EMCEE]")
# ============================
#       BROKEN FUNCTION
# ============================

def mcmc_multi(file_path: str, n_max: int, n_iter: int, n_burnin: int, n_walkers: int=16, progress: bool=True) -> None:
    """ Run MCMC EMCEE for all coincidences up to n_max using multiprocessing """
    nants        = np.load(os.path.join(file_path, "co_nants.npy"))
    n_coinc      = len(nants)
    n_to_process = min(n_coinc, n_max)
    n_cpus       = cpu_count()
    batch_size   = n_cpus * 4

    time_multi_start = time.time()
    # Multiprocessing pool
    for batch_start in tqdm(range(0, n_to_process, batch_size)):
        if progress:
            batch_start_time = time.time()
        batch_end = min(batch_start + batch_size, n_to_process)

        args_list = [(file_path, coinc_idx, n_iter, n_burnin, n_walkers) for coinc_idx in range(batch_start, batch_end)] 

        with Pool(processes=n_cpus) as pool:
            batch_results = pool.map(single_idx_worker, args_list)

        for coinc_idx, modes in batch_results:
            save_modes(file_path, coinc_idx, modes, n_total=n_to_process)
            del modes, coinc_idx # free memory

        del batch_results, args_list # free memory

        if progress:
            batch_end_time = time.time()
            print(f"\nBatch {batch_start} to {batch_end-1} processed in {batch_end_time - batch_start_time:.2f} seconds.")
        
    time_multi_end = time.time()
    print(f"\n\n[Processed {n_to_process} coincidences in MCMC EMCEE with multiprocessing in {time_multi_end - time_multi_start:.2f} seconds]")

def single_idx_worker(args: Tuple) -> Tuple[int, np.ndarray, np.ndarray]:
    file_path, coinc_idx, n_iter, n_burnin, n_walkers = args
    
    # Load data and run MCMC
    params, data = load_shower_data(file_path, coinc_idx)
    modes = mcmc_single_idx(params, data,
                                n_steps=n_iter,
                                n_burnin=n_burnin,
                                n_walkers=n_walkers,
                                progress=True, # progress bar not needed in multiprocessing
                                return_mode=True) # force return_mode to True when multiprocessing

    print(f"Finished MCMC for coincidence idx {coinc_idx}...")
    del params, data # free memory
    return (coinc_idx, modes)


# ---------- Plotting & Saving Results ----------

def modes_finder(samples: np.ndarray) -> np.ndarray:
    """ Trouve les modes des distributions des samples MCMC """
    modes = []
    for i in range(samples.shape[1]):
        hist, bin_edges = np.histogram(samples[:, i], bins=50)
        max_bin_index = np.argmax(hist)
        mode_argmax = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1])
        modes.append(mode_argmax)
    return np.array(modes)

def save_modes(file_path: str, coinc_idx: int, modes: np.ndarray, n_total: int) -> None:
    """
    Sauvegarde des modes des résultats MCMC EMCEE dans un fichier HDF5 (pour garder de la place).
    """
    h5_file = os.path.join(file_path, "mcmc_modes_results.h5")
    if coinc_idx == 0 and os.path.exists(h5_file):
        os.remove(h5_file)
    
    # Créer ou ouvrir le fichier HDF5
    mode = 'a' if os.path.exists(h5_file) else 'w'
    
    with h5py.File(h5_file, mode) as f:
        # Créer le groupe principal s'il n'existe pas
        if 'mcmc_emcee_results' not in f:
            grp = f.create_group('mcmc_emcee_results')
            grp.attrs['n_total'] = n_total
        else:
            grp = f['mcmc_emcee_results']
        
        coinc_grp_name = f'coinc_{coinc_idx:05d}'
        
        if coinc_grp_name in grp:
            del grp[coinc_grp_name]
        
        coinc_grp = grp.create_group(coinc_grp_name)
        
        # Sauvegarder les modes
        coinc_grp.create_dataset('modes', data=modes, compression='gzip')
        coinc_grp.attrs['n_params'] = len(modes)
        
    print(f"Coincidence {coinc_idx} modes sauvegardée dans le fichier {h5_file}")

def save_samples_res(file_path: str, coinc_idx: int, samples: np.ndarray, log_prob: np.ndarray, n_total: int) -> None:
    """
    Sauvegarde des résultats MCMC EMCEE dans un fichier HDF5.
    """
    h5_file = os.path.join(file_path, "mcmc_sample_results.h5")
    if coinc_idx == 0 and os.path.exists(h5_file):
        os.remove(h5_file)

    # Créer ou ouvrir le fichier HDF5
    mode = 'a' if os.path.exists(h5_file) else 'w'
    
    with h5py.File(h5_file, mode) as f:
        # Créer le groupe principal s'il n'existe pas
        if 'mcmc_emcee_results' not in f:
            grp = f.create_group('mcmc_emcee_results')
            grp.attrs['n_total'] = n_total
        else:
            grp = f['mcmc_emcee_results']
        
        coinc_grp_name = f'coinc_{coinc_idx:05d}'
        
        if coinc_grp_name in grp:
            del grp[coinc_grp_name]
        
        coinc_grp = grp.create_group(coinc_grp_name)
        
        # Sauvegarder les samples
        coinc_grp.create_dataset('samples', data=samples, compression='gzip')
        coinc_grp.create_dataset('log_prob', data=log_prob, compression='gzip')
        coinc_grp.attrs['n_samples'] = samples.shape[0]
        coinc_grp.attrs['n_params'] = samples.shape[1]
        
    print(f"Coincidence {coinc_idx} sauvegardée dans le fichier {h5_file}")

def corner_plot(samples: np.ndarray, save_path: str, coinc_idx: int) -> None:
    """ Corner plot of all parameters"""
    # All parameters
    figure1 = corner.corner(
        samples, 
        labels=["Alpha (def)", "Beta (deg)", "Rxmax (m)", "t0 (s)", "Theta (deg)", "Phi (deg)", "Delta Omega (m)", "Amplitude"],
        show_titles=True, 
        title_fmt=".2f", 
        quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 12},
        color='#e74c3c',                # Rouge moderne
        hist_kwargs={
            'color': '#e74c3c',
            'edgecolor': '#c0392b',
            'linewidth': 1.5,
            'alpha': 0.7
        },
        contour_kwargs={
            'colors': ['#e74c3c', '#c0392b', '#a93226'],  # Dégradé de rouge
            'linewidths': [1.0, 1.5, 2.0]
        },
        scatter_kwargs={
            'alpha': 0.2,
            's': 2,
            'color': '#e74c3c',
            'rasterized': True
        },
        smooth=1.0,
        fill_contours=True,
        levels=(0.68, 0.95),
    )

    figure1.savefig(
        os.path.join(save_path, f"corner_coinc_{coinc_idx:05d}.png"),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close(figure1)
    print(f"Corner plot saved for coincidence {coinc_idx} at {save_path}")


# ---------- Main Execution ----------
def main() -> None:
    # restructure data for efficient loading
    if not os.path.exists(os.path.join(file_path, "SWF_array.npy")) or not os.path.exists(os.path.join(file_path, "ADF_array.npy")):
        restructure_data(file_path)

    # ------------- TESTS -------------
    # Test a run for a single coincidence idx
    if args.test:
        print("Running MCMC EMCEE test for a single coincidence index...\n")

        coinc_idx    = 11 # Example coincidence index for testing
        params, data = load_shower_data(file_path, coinc_idx)
        samples, _   = mcmc_single_idx(params, data, n_steps=n_iter_v, n_burnin=n_burnin_v, n_walkers=4, progress=True, thinning=False, return_mode=False)
        modes = modes_finder(samples)
        print(f"\nMCMC EMCEE modes for coincidence {coinc_idx}: {modes}")
        
        corner_plot(samples, save_path=file_path, coinc_idx=coinc_idx)
        print(f"Corner plot saved for coincidence {coinc_idx} at {file_path}")

        print("\nMCMC EMCEE test completed!")
        return

    # ------------- SWF only -------------
    # Run on the SWF only to check convergence and that it aligns with fit on SWF
    elif args.swf:
        print("Running MCMC EMCEE for SWF parameters only...\n")
        mcmc_sequential_swf(file_path, n_max=30, n_iter=n_iter_v, n_burnin=n_burnin_v, n_walkers=16, progress=True, thinning=False)

    # ------------- Full MCMC -------------
    # Real runs for multiple coincidence indices to retrieve the means and stds, to have results quickly and efficiently
    elif args.multi:
        print("Running MCMC EMCEE with multiprocessing for multiple coincidence indices...\n")
        mcmc_multi(file_path, n_max=n_max, n_iter=n_iter_v, n_burnin=n_burnin_v, n_walkers=32, progress=True)

    # ------------- Full MCMC Sequential -------------
    # Used to get the full samples and doi statsitical analysis afterwards, produces corner plot, logprob values and VERY large files
    else:
        print("Running MCMC EMCEE for multiple coincidence indices...\n")
        mcmc_sequential(file_path, n_max=n_max, n_iter=n_iter_v, n_burnin=n_burnin_v, n_walkers=32, progress=True, thinning=True, start_idx=start_idx_a, corner_plot_bool=True)

    print("\nMCMC processing completed!")

if __name__ == "__main__":
    main()