# Imports
from importlib.metadata import files
import os
import jax
import time
import argparse
import numpy           as np
import pandas          as pd
import jax.numpy       as jnp
import multiprocessing as mp
import wavefronts.energy_jax    as ej
import wavefronts.params_config as pr
import wavefronts.loader_txt    as lo
import wavefronts.Gram_Recons   as gr
import wavefronts.compute_basic as cb
from tqdm            import tqdm
from typing          import Tuple
from iminuit         import minimize
from scipy.optimize  import differential_evolution
from wavefronts.wavefronts_SEB import *

# argparse setup
parser = argparse.ArgumentParser(description='Cramer-Rao Bound computation for radio-detected air showers')
parser.add_argument('-nmax'    , type=int, default=None,      help='Maximum number of coincidences to process')
parser.add_argument('-filepath', type=str, default='test_NJ', help='Path to the input data files') # other exemple 'test_AN3'
parser.add_argument('-multi'   , action='store_false',        help='Enable multiprocessing for SWF reconstructions. Default is enabled')
parser.add_argument('-test'    , action='store_true',         help='Run all computes in test mode with a small dataset')
parser.add_argument('-all'     , action='store_true',         help='Run all reconstructions and CRB computations')
parser.add_argument('-verbose' , action='store_true',         help='Enable verbose output during reconstructions')
parser.add_argument('-swf',      action='store_true',         help='Force computation of SWF reconstruction')
parser.add_argument('-adf',      action='store_true',         help='Force computation of ADF reconstruction')
parser.add_argument('-crb',      action='store_true',         help='Force computation of CRB')
parser.add_argument('-energy',   action='store_true',         help='Force computation of energy reconstruction')
parser.add_argument('-grammage', action='store_true',         help='Force computation of grammage reconstruction')
parser.add_argument('-old',      action='store_true',         help='Run the old dataframe builder instead of the new one (for Marion\'s files)')
parser.add_argument('-build',    action='store_true',         help='Only build the .npy files from the input text files, without running reconstructions or CRB computations')
args = parser.parse_args()

# Numpy compatibility 
if not hasattr(np, 'infty'):
    np.infty = np.inf



# ======================= Miscellaneous ======================== #

def add_df_columns(df: pd.DataFrame, events_ids: np.ndarray, SWF_res: np.ndarray=None, SWF_loss: np.ndarray=None, ADF_res: np.ndarray=None, ADF_loss: np.ndarray=None, CRB_res: np.ndarray=None, CRB_ADF_only: np.ndarray=None, energies: np.ndarray=None, energies_uncertainty: np.ndarray=None, grammages: np.ndarray=None, xcore: np.ndarray=None) -> pd.DataFrame:
    """ Add columns to the DataFrame with reconstruction results
    Inputs:
        df: pandas DataFrame
        events_ids: array containing event indices for each coincidence
        SWF_res: array containing SWF reconstruction results
        SWF_loss: array containing SWF loss values
        ADF_res: array containing ADF reconstruction results
        ADF_loss: array containing ADF loss values
        CRB_res: array containing CRB results
        CRB_ADF_only: array containing CRB results for ADF only
        energies: array containing reconstructed energies
        energies_uncertainty: array containing uncertainties on reconstructed energies
        grammages: array containing reconstructed grammages
        xcore: array containing Xcore coordinates
    Outputs:
        df: pandas DataFrame with added columns
    """
    def assign_columns(df, data, col_names, events_ids):
        """Assign or replace columns in df, aligned on event_idx."""
        df_extra = pd.DataFrame(data, columns=col_names)
        df_extra['event_idx'] = events_ids
        df_extra = df_extra.set_index('event_idx')
        for col in col_names:
            df[col] = df['event_idx'].map(df_extra[col])
        return df


    if SWF_res is not None:
        df = assign_columns(df, SWF_res, ['recons_alpha', 'recons_beta', 'recons_rxmax', 'recons_t0'], events_ids)

    if SWF_loss is not None:
        df = assign_columns(df, SWF_loss.reshape(-1,1), ['SWF_loss'], events_ids)

    if ADF_res is not None:
        df = assign_columns(df, ADF_res, ['recons_theta', 'recons_phi', 'recons_delta_omega', 'recons_amplitude'], events_ids)

    if ADF_loss is not None:
        df = assign_columns(df, ADF_loss.reshape(-1,1), ['ADF_loss'], events_ids)

    if CRB_res is not None:
        df = assign_columns(df, CRB_res, ['stds_alpha', 'stds_beta', 'stds_rxmax', 'stds_t0', 'stds_theta', 'stds_phi', 'stds_delta_omega', 'stds_amplitude'], events_ids)

    if CRB_ADF_only is not None:
        df = assign_columns(df, CRB_ADF_only, ['stds_theta_adf', 'stds_phi_adf', 'stds_delta_omega_adf', 'stds_amplitude_adf'], events_ids)

    if energies is not None and energies_uncertainty is not None:
        df = assign_columns(df, np.stack([energies, energies_uncertainty], axis=1), ['recons_energy', 'recons_energy_uncertainty'], events_ids)

    if grammages is not None:
        df = assign_columns(df, np.array(grammages).reshape(-1,1), ['recons_grammage'], events_ids)

    if xcore is not None:
        df = assign_columns(df, xcore, ['x_core', 'y_core', 'z_core'], events_ids)
        df = assign_columns(df, np.sqrt(xcore[:,0]**2 + xcore[:,1]**2).reshape(-1,1), ['dist_xcore'], events_ids)

    return df



# ============================ PWF ============================ #

def PWF_recons(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_time_array: np.ndarray, n_max: int, verbose: bool=False) -> np.ndarray:
    """ PWF reconstruction for all coincidences
    Inputs:
        ncoincs: number of coincidences
        nants: array of number of antennas per coincidence
        antenna_coords_array: array of antenna coordinates per coincidence
        peak_time_array: array of peak times per coincidence
    Outputs:
        PWF_res: dictionary containing PWF reconstruction results
    """

    n_to_process = n_max

    # t0 = time.time()
    rad2deg = 180.0 / np.pi
    PWF_res = np.zeros((n_to_process, 2))  # theta, phi in degrees

    for i in range(n_to_process):
        try:
            # print(f"Value of tants for coincidence {i}: {peak_time_array[i,:nants[i]]}")
            theta_PWF_rad, phi_PWF_rad = PWF_minimize_alternate_loss_norm(antenna_coords_array[i,:nants[i]], peak_time_array[i,:nants[i]], verbose)
            theta_PWF_deg = rad2deg * theta_PWF_rad
            phi_PWF_deg   = rad2deg * phi_PWF_rad
            PWF_res[i,0]  = theta_PWF_deg
            PWF_res[i,1]  = phi_PWF_deg
            if verbose and 1>2: print(f"Results are : theta_PWF = {theta_PWF_deg:.3f}°, phi_PWF = {phi_PWF_deg:.3f}°")

        except Exception as e:
            if verbose : print(f"PWF reconstruction failed for coincidence {i} with error: {e}")
            PWF_res[i,0] = np.nan
            PWF_res[i,1] = np.nan

    # print(f"[{time.time()-t0:.3f}s] Plane Wave Fit reconstruction done for {n_to_process} coincidences")
    return PWF_res



# ============================ SWF ============================ #

def SWF_recons(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_time_array: np.ndarray, PWF_res: np.ndarray, verbose: bool=False, n_max: int=None, event_type: str='EAS'):
        """ SWF reconstruction for all coincidences
        Inputs:
            ncoincs: number of coincidences
            nants: array of number of antennas per coincidence
            antenna_coords_array: array of antenna coordinates per coincidence
            peak_time_array: array of peak times per coincidence
            PWF_res: dictionary containing PWF reconstruction results
        Outputs:
            SWF_res: dictionary containing SWF reconstruction results
        """

        n_to_process = ncoincs if n_max is None else min(ncoincs, n_max)
        t0 = time.time()
        deg2rad = np.pi / 180.0
        SWF_res = np.zeros((n_to_process, 4))  # alpha, beta in degrees, rxmax, t_0
        SWF_losses = np.zeros(n_to_process) # SWF loss values   

        for i in tqdm(range(n_to_process), desc='SWF in progress...'):
            try:
                alpha_PWF_rad = PWF_res[i,0] * deg2rad # we use theta and phi from PWF as initial guesses for alpha and beta
                beta_PWF_rad  = PWF_res[i,1] * deg2rad # they should not be too far, but not quite exactly the same

                _, alpha_SWF_deg, beta_SWF_deg, rxmax_SWF, t0_SWF, swf_loss = SWF_single_recon(i, alpha_PWF_rad, beta_PWF_rad, antenna_coords_array[i,:nants[i]], peak_time_array[i,:nants[i]], verbose, event_type)

                SWF_res[i,0] = alpha_SWF_deg
                SWF_res[i,1] = beta_SWF_deg
                SWF_res[i,2] = rxmax_SWF
                SWF_res[i,3] = t0_SWF
                SWF_losses[i] = swf_loss

            except Exception as e:
                if verbose : print(f"SWF reconstruction failed for coincidence {i} with error: {e}")
                SWF_res[i,:] = np.nan
        
        print(f"\n[{time.time()-t0:.3f}s] Spherical Wave Fit reconstruction done for {n_to_process} coincidences")

        return SWF_res, SWF_losses

def SWF_single_recon(i: int, alpha_PWF_rad: float, beta_PWF_rad: float, ant_coords: np.ndarray, peak_time_arr: np.ndarray, verbose: bool, event_type: bool='EAS') -> Tuple[int, float, float, float, float]:
    rad2deg = 180.0 / np.pi
    deg2rad = np.pi / 180.0

    try:
        # Définition des bornes d'optimisation
        if event_type == 'EAS':
            alpha_bounds = [alpha_PWF_rad - 3*deg2rad, alpha_PWF_rad + 3*deg2rad]
            beta_bounds = [beta_PWF_rad - 3*deg2rad, beta_PWF_rad + 3*deg2rad]
            rxmax_bounds = pr.bounds[2]
            t0_bounds = pr.bounds[3]
            
            bounds = np.array([alpha_bounds, beta_bounds, rxmax_bounds, t0_bounds], dtype=np.float64)

        else:
            # Cas 'wide angle' : utilisation des bornes par défaut si non précisé
            bounds = pr.bounds

        # Initial guess
        PWF_guess = np.array(bounds, dtype=np.float64).mean(axis=1)

        args = (ant_coords, peak_time_arr, True) # if true returns chi2/ndof
        
        resu = differential_evolution(SWF_loss, bounds, args=args, strategy='best1bin', maxiter=3000, seed=42, tol=1e-6, mutation=(0.5, 1), recombination=0.7, x0=PWF_guess)

        alpha_SWF_deg = resu.x[0] * rad2deg
        beta_SWF_rad  = resu.x[1] % (2 * np.pi) # careful with modulo 2pi
        beta_SWF_deg  = beta_SWF_rad * rad2deg
        rxmax_SWF     = resu.x[2]

        if verbose :
            print(f"SWF initial guess for coincidence {i}:")
            print(f"  alpha : {PWF_guess[0]*rad2deg:10.2f}°   →   {alpha_SWF_deg:10.2f}°")
            print(f"  beta  : {PWF_guess[1]*rad2deg:10.2f}°   →   {beta_SWF_deg:10.2f}° (Corrigé Modulo 360)")
            print(f"  rxmax : {PWF_guess[2]:10.2f}m   →   {rxmax_SWF:10.2f}m")

        return (i, alpha_SWF_deg, beta_SWF_deg, rxmax_SWF, resu.x[3], resu.fun)
    except Exception as e:
        if verbose : print(f"SWF reconstruction failed for coincidence {i} with error: {e}")
        return (i, np.nan, np.nan, np.nan, np.nan, np.nan)

def worker_function(args: Tuple) -> Tuple[int, float, float, float, float]:
    """Fonction wrapper pour multiprocessing (doit être picklable)"""
    i, alpha_PWF_rad, beta_PWF_rad, ant_coords, peak_time_arr, verbose, event_type = args
    return SWF_single_recon(i, alpha_PWF_rad, beta_PWF_rad, ant_coords, 
                            peak_time_arr, verbose, event_type)

def SWF_recons_mp(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_time_array: np.ndarray, PWF_res: np.ndarray, verbose: bool=False, event_type: str='EAS', n_max: int=None) -> np.ndarray:
    """
    SWF reconstruction with multiprocessing.
    
    Uses min(128, max(1, CPU_count-1)) processes.
    Processes all data at once and saves to NPY format only.
    
    Parameters:
    -----------
    ncoincs : int - Total number of coincidences
    nants : array - Number of antennas per coincidence
    antenna_coords_array : array - Antenna coordinates [ncoincs, max_nants, 3]
    peak_time_array : array - Peak times [ncoincs, max_nants]
    PWF_res : array - PWF results [ncoincs, n_params], alpha/beta at columns 0/1
    n_max : int - Max coincidences to process
    verbose : bool - Detailed output
    event_type : str - 'EAS' or other
    
    Returns:
    --------
    SWF_res : array [n_to_process, 4] - [alpha_deg, beta_deg, rxmax, t0]
    SWF_losses : array [n_to_process] - SWF loss values
    """
    
    deg2rad = np.pi / 180.0
    n_to_process = ncoincs if n_max is None else min(ncoincs, n_max)
    
    # Setup multiprocessing: min(128, max(1, CPU_count-1))
    cpu_count = mp.cpu_count()
    n_processes = max(1, min(128, cpu_count - 1))
    t1 = time.time()
    
    # Prepare arguments (will be deleted after use)
    args_list = []
    for i in range(n_to_process):
        alpha_PWF_rad = PWF_res[i, 0] * deg2rad
        beta_PWF_rad = PWF_res[i, 1] * deg2rad
        ant_coords = antenna_coords_array[i, :nants[i]].copy()
        peak_time_arr = peak_time_array[i, :nants[i]].copy()
        
        args_list.append((i, alpha_PWF_rad, beta_PWF_rad, ant_coords, 
                        peak_time_arr, verbose, event_type))
    
    # Parallel processing with progress bar
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(worker_function, args_list),
            total=n_to_process,
            desc="SWF Reconstruction",
            unit="coinc"
        ))
    
    # Free memory immediately
    del args_list
    
    # Collect results into array
    SWF_res = np.zeros((n_to_process, 4), dtype=np.float64)
    SWF_losses = np.zeros(n_to_process, dtype=np.float64)
    for result in results:
        idx, alpha, beta, rxmax, t0, swf_loss = result
        SWF_res[idx] = [alpha, beta, rxmax, t0]
        SWF_losses[idx] = swf_loss
    
    # Free memory
    del results
    
    print(f"\n[{time.time()-t1:.3f}s] SWF reconstruction done for {n_to_process} coincidences")
    
    return SWF_res, SWF_losses



# ============================ ADF ============================ #

def ADF_recons(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_amp_array: np.ndarray, PWF_res: np.ndarray, SWF_res: np.ndarray, B_vecs: np.ndarray, verbose: bool=False, n_max: int=None) -> np.ndarray:
    """ ADF reconstruction for all coincidences 
    Inputs:
        ncoincs: number of coincidences
        nants: array of number of antennas per coincidence
        antenna_coords_array: array of antenna coordinates per coincidence
        peak_amp_array: array of peak amplitudes per coincidence
        PWF_res: dictionary containing PWF reconstruction results
        SWF_res: dictionary containing SWF reconstruction results
        B_vecs: array of magnetic field vectors
        verbose: boolean indicating whether to print detailed output
        n_max: maximum number of coincidences to process

    Outputs:
        ADF_res: dictionary containing ADF reconstruction results """
    
    n_to_process = ncoincs if n_max is None else min(ncoincs, n_max)
    t0 = time.time()
    r2d, d2r = 180.0/np.pi, np.pi/180.0 # degrees to radians conversion factors
    
    # Vectorized preprocessing
    theta_PWF_rad = PWF_res[:n_to_process,0] * d2r
    phi_PWF_rad   = PWF_res[:n_to_process,1] * d2r
    alpha_PWF_rad = SWF_res[:n_to_process,0] * d2r
    beta_PWF_rad  = SWF_res[:n_to_process,1] * d2r
    rx_max        = SWF_res[:n_to_process,2]
    B_vec         = B_vecs[:n_to_process]
    
    ADF_res = np.zeros((n_to_process, 4))
    ADF_losses = np.zeros(n_to_process)
    
    for i in tqdm(range(n_to_process), desc='ADF in progress...'):

        i, theta_deg, phi_deg, dw, Amp, loss = ADF_single_recon(i, theta_PWF_rad[i], phi_PWF_rad[i], rx_max[i], alpha_PWF_rad[i], beta_PWF_rad[i], antenna_coords_array[i,:nants[i]], peak_amp_array[i,:nants[i]], B_vec[i], verbose)

        ADF_res[i,0] = theta_deg
        ADF_res[i,1] = phi_deg
        ADF_res[i,2] = dw
        ADF_res[i,3] = Amp
        ADF_losses[i] = loss

    print(f"[{time.time()-t0:.3f}s] ADF done for {n_to_process} coincidences")

    return ADF_res, ADF_losses

def ADF_single_recon(i: int, theta_PWF_rad: float, phi_PWF_rad: float, rx_max: float, alpha_rad: float, beta_rad: float, ant_coords: np.ndarray, peak_amp_arr: np.ndarray, B_vec: np.ndarray, verbose: bool=False) -> Tuple[int, float, float, float, float]:
    """Single ADF reconstruction for one coincidence, ca=cos(alpha_PWF), sa=sin(alpha_PWF), cb=cos(beta_PWF), sb=sin(beta_PWF)"""
    r2d, d2r = 180.0/np.pi, np.pi/180.0
    
    try:
        # Xmax position
        Xmax = cb.build_Xsource_np(alpha_rad, beta_rad, rx_max)

        # Bounds and initial guess
        angle_pm = 3*d2r
        bounds   = np.array([[theta_PWF_rad-angle_pm, theta_PWF_rad+angle_pm], # theta bounds
                           [phi_PWF_rad-angle_pm, phi_PWF_rad+angle_pm],     # phi bounds
                           pr.bounds[6], pr.bounds[7]],  # dw and Amp bounds
                           dtype=np.float64)     

        max_idx = peak_amp_arr.argmax()
        Amp_guess = np.linalg.norm(ant_coords[max_idx]-Xmax)*peak_amp_arr[max_idx] # propagation of 1/r to highest amplitude antenna => Amp guess = r*Amp_max
        initial_guess = np.array([theta_PWF_rad, phi_PWF_rad, 5, Amp_guess], dtype=np.float64)
        
        # Optimization
        res = minimize(ADF_loss, initial_guess, bounds=bounds, 
                      args=(peak_amp_arr, ant_coords, Xmax, False, B_vec), 
                      method='migrad', tol=1e-5)
        
        if verbose:
            print(f"Xmax {i}: X={Xmax[0]:.2e}, Y={Xmax[1]:.2e}, Z={Xmax[2]-pr.groundAltitude:.2e}")
            print(f"Xmax distance to (0,0,0) : {np.linalg.norm(Xmax)/1e3:.2e} km")
            print( f"  θ  : {initial_guess[0]*r2d:10.2f}°   →   {res.x[0]*r2d:10.2f}°, with bounds [{bounds[0,0]*r2d:.2f}°, {bounds[0,1]*r2d:.2f}°], intial guess {initial_guess[0]*r2d:.2f}°, alpha_PWF {alpha_rad*r2d:.2f}°")
            print( f"  φ  : {initial_guess[1]*r2d:10.2f}°   →   {res.x[1]*r2d:10.2f}°, with bounds [{bounds[1,0]*r2d:.2f}°, {bounds[1,1]*r2d:.2f}°], intial guess {initial_guess[1]*r2d:.2f}°, beta_PWF {beta_rad*r2d:.2f}°")
            print(rf"  dw : {initial_guess[2]:10.2f}    →   {res.x[2]:10.2f} , with bounds [{bounds[2,0]:.2f}, {bounds[2,1]:.2f}]")
            print( f"  A  : {initial_guess[3]:10.2e}    →   {res.x[3]:10.2e} , with bounds [{bounds[3,0]:.2e}, {bounds[3,1]:.2e}]")
            print(f"Loss : {res.fun:.2e}")
        
        return (i, res.x[0]*r2d, (res.x[1] % (2*np.pi))*r2d, res.x[2], res.x[3], res.fun) # careful with modulo 2pi
    
    except Exception as e:
        if verbose:
            print(f"ADF reconstruction failed for coincidence {i} with error: {e}")
            print(e)
        return (i, np.nan, np.nan, np.nan, np.nan, np.nan)

def worker_function_adf(args: Tuple) -> Tuple[int, float, float, float, float]:
    """Wrapper for multiprocessing"""
    i, th, ph, rx, al, be, B_vec, ant_coords, peak_amp_arr, verbose = args
    return ADF_single_recon(i, th, ph, rx, al, be, ant_coords, peak_amp_arr, B_vec, verbose)

def ADF_recons_mp(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_amp_array: np.ndarray, PWF_res: np.ndarray, SWF_res: np.ndarray, B_vecs: np.ndarray, verbose: bool=False, n_max: int=None) -> np.ndarray:
    """
    ADF reconstruction with multiprocessing.
    
    Uses min(128, max(1, CPU_count-1)) processes.
    Processes all data at once and saves to NPY format only.
    
    Parameters:
    -----------
    ncoincs : int - Total number of coincidences
    nants : array - Number of antennas per coincidence
    antenna_coords_array : array - Antenna coordinates [ncoincs, max_nants, 3]
    peak_amp_array : array - Peak amplitudes [ncoincs, max_nants]
    PWF_res : array - PWF results [ncoincs, n_params], theta/phi at columns 0/1
    SWF_res : array - SWF results [ncoincs, n_params], rxmax at column 2
    n_max : int - Max coincidences to process
    groundAltitude : float - Ground altitude
    B_vecs : array - Magnetic field vectors [ncoincs, 3]
    verbose : bool - Detailed output
    
    Returns:
    --------
    ADF_res : array [n_to_process, 4] - [theta_deg, phi_deg, dw, Amp]
    """
    
    n_to_process = min(ncoincs, n_max) if n_max is not None else ncoincs
    r2d, d2r = 180.0/np.pi, np.pi/180.0
    
    # Setup multiprocessing: min(128, max(1, CPU_count-1))
    cpu_count = mp.cpu_count()
    n_processes = max(1, min(128, cpu_count - 1))
    t0 = time.time()
    
    # Vectorized preprocessing
    th = PWF_res[:n_to_process, 0] * d2r
    ph = PWF_res[:n_to_process, 1] * d2r
    al = SWF_res[:n_to_process, 0] * d2r
    be = SWF_res[:n_to_process, 1] * d2r
    rx = SWF_res[:n_to_process, 2]
    Bv = B_vecs[:n_to_process]

    # Prepare arguments (will be deleted after use)
    args_list = []
    for i in range(n_to_process):
        ant_coords = antenna_coords_array[i, :nants[i]].copy()
        peak_amp_arr = peak_amp_array[i, :nants[i]].copy()
        
        args_list.append((i, th[i], ph[i], rx[i], al[i], be[i], Bv[i],
                        ant_coords, peak_amp_arr, verbose))
    
    # Free memory from preprocessed arrays
    del th, ph, al, be, rx
    
    # Parallel processing with progress bar
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(worker_function_adf, args_list),
            total=n_to_process,
            desc="ADF Reconstruction",
            unit="coinc"
        ))
    
    # Free memory immediately
    del args_list
    
    # Collect results into array
    ADF_res = np.zeros((n_to_process, 4), dtype=np.float64)
    ADF_losses = np.zeros(n_to_process, dtype=np.float64)
    for result in results:
        idx, theta, phi, dw, amp, loss = result
        ADF_res[idx] = [theta, phi, dw, amp]
        ADF_losses[idx] = loss
    
    # Free memory
    del results
    
    print(f"\n[{time.time()-t0:.3f}s] ADF reconstruction done for {n_to_process} coincidences")
    return ADF_res, ADF_losses



# ======================= CRB of ADF + SWF ======================= #

def ADF_SWF_CRB(ncoincs: int, nants: np.ndarray, antennas_coords: np.ndarray, SWF_res: np.ndarray, ADF_res: np.ndarray, filepath: str, B_vecs: np.ndarray, n_max: int=None, verbose: bool=False) -> np.ndarray:

    """ Function calculating the Cramér-Rao Bound for the joint ADF + SWF reconstruction
    Inputs:
        ncoincs: number of coincidences
        nants: array of number of antennas per coincidence
        antennas_coords: array of antenna coordinates per coincidence
        SWF_res: dictionary containing SWF reconstruction results
        ADF_res: dictionary containing ADF reconstruction results
        B_vecs: array of magnetic field vectors
        filepath: path to the input file
        n_max: maximum number of coincidences to process
        verbose: boolean for verbosity
    Outputs:
        stds: array of standard deviations for each parameter per coincidence"""

    t0           = time.time()                 # Début du timer
    n_to_process = ncoincs if n_max is None else  min(ncoincs, n_max) # Nombre de coïncidences à traiter
    stds         = np.zeros((n_to_process, 8)) # stds pré-allocation
    deg2rad      = np.pi / 180.0
    rad2deg      = 180.0 / np.pi
    cpt          = 0

    # Determine noise floor and amplitude uncertainty based on file path
    is_efield = 'efield' in filepath
    is_gp300 = 'GP300' in filepath
    is_gp289 = 'GP289' in filepath
    is_nj_adc = '-NJ_adc' in filepath
    is_an_adc = '-AN_adc' in filepath

    if is_efield:
        min_amplitude = 1e-3 # 1e-3 µV/m, minimal increment of values
        jitter_time   = 0.5e-9 # 0.5 ns, typical time resolution for electric field measurements
        galactic_noise_floor = 0.0
        amplitude_uncertainty = 0.0

    elif is_gp300:  # ZHAireS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_nj_adc:
            jitter_time = 0.5e-9
            galactic_noise_floor = 0.0
            amplitude_uncertainty = 0.0
        elif is_an_adc:
            jitter_time = 5e-9
            galactic_noise_floor = 15.0
            amplitude_uncertainty = 0.075
        else:
            jitter_time = 5e-9
            galactic_noise_floor = 4.0
            amplitude_uncertainty = 0.075

    elif is_gp289:  # ZHAireS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_nj_adc:
            jitter_time = 0.5e-9
            galactic_noise_floor = 0.0
            amplitude_uncertainty = 0.0
        elif is_an_adc:
            jitter_time = 5e-9
            galactic_noise_floor = 12.0
            amplitude_uncertainty = 0.075
        else:
            jitter_time = 5e-9
            galactic_noise_floor = 5.0
            amplitude_uncertainty = 0.075

    else:  # CoREAS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_an_adc:
            jitter_time = 5e-9
            galactic_noise_floor = 10.0
            amplitude_uncertainty = 0.075

    for current_recons in tqdm(range(n_to_process), desc='ADF + SWF CRB computing...'):
        n_ants = nants[current_recons]
        ant_coords = antennas_coords[current_recons, :n_ants] # Coords
        
        swf = SWF_res[current_recons]
        adf = ADF_res[current_recons]

        swf_rad = swf.copy()
        swf_rad[0] *= deg2rad
        swf_rad[1] *= deg2rad
        adf_rad = adf.copy()
        adf_rad[0] *= deg2rad
        adf_rad[1] *= deg2rad

        Xsource = cb.build_Xsource_np(swf_rad[0], swf_rad[1], swf_rad[2])
        
        params = np.hstack((swf_rad, adf_rad))
        B_vec = B_vecs[current_recons]
        fisher_mat = np.zeros((8,8))

        # Calcul des dérivées pour tous les paramètres
        h = 1e-6 * (np.abs(params)) ; h[3] = 1e-9
        derivates_ampl = np.zeros((n_ants, 8))
        derivates_time = np.zeros((n_ants, 8))

        # Calcul de la dérivée pour chaque antenne, par rapport à chaque paramètre
        for i in range(8):
            # Perturbations symétriques
            params_plus  = params.copy() ; params_plus[i]  += h[i]
            params_minus = params.copy() ; params_minus[i] -= h[i]
            
            # Dérivée par différences finies
            swf_params_plus  = params_plus[:4]
            adf_params_plus  = params_plus[4:]
            swf_params_minus = params_minus[:4]
            adf_params_minus = params_minus[4:]

            # Reconstruction of Xmax for perturbed SWF parameters
            # Plus
            X_max_plus = cb.build_Xsource_np(swf_params_plus[0], swf_params_plus[1], swf_params_plus[2])
            X_max_minus = cb.build_Xsource_np(swf_params_minus[0], swf_params_minus[1], swf_params_minus[2])
            
            pred_plus_ampl  = ADF_3D_model(adf_params_plus, ant_coords, X_max_plus, B_vec) # in mV
            pred_plus_time  = SWF_model(swf_params_plus, ant_coords) # in s

            pred_minus_ampl  = ADF_3D_model(adf_params_minus, ant_coords, X_max_minus, B_vec) # in mV
            pred_minus_time  = SWF_model(swf_params_minus, ant_coords) # in s
            
            # Dérivée
            derivates_ampl[:, i] = (pred_plus_ampl - pred_minus_ampl) / (2 * h[i])
            derivates_time[:, i] = (pred_plus_time - pred_minus_time) / (2 * h[i])
        
        sigma_amp = amplitude_uncertainty * abs(ADF_3D_model(adf_rad, ant_coords, Xsource, B_vec))
        sigma_amp = [(sigma_amp[i]**2 + galactic_noise_floor**2 + min_amplitude**2)**0.5 for i in range(n_ants)] 
        sigma_amp = np.array(sigma_amp)
        sigma_amp = np.where(sigma_amp == 0, 1, sigma_amp) # Avoid division by zero
        sigma_time = (jitter_time) # Fixed time uncertainty in s
            
        for k in range(n_ants):
            fisher_mat += np.outer(derivates_ampl[k,:], derivates_ampl[k,:]) / (sigma_amp[k]**2)
            fisher_mat += np.outer(derivates_time[k,:], derivates_time[k,:]) / (sigma_time**2)

        try:
            cov_mat = np.linalg.inv(fisher_mat)
            stds[current_recons, :] = np.sqrt(np.diag(cov_mat)) # Écarts-types
            if np.any(np.isnan(stds[current_recons, :])) or np.any(np.isinf(stds[current_recons, :])):
                if verbose and 0<1:
                    print(f"Fisher matrix inversion NaN of Inf for coinc {current_recons}.")
                    # print(fisher_mat)
                stds[current_recons, :] = np.array([np.nan]*8)
                cpt += 1
        except np.linalg.LinAlgError:
            if verbose and 0<1:
                print(f"Fisher matrix is singular for coinc {current_recons}.")
                print(fisher_mat)
            stds[current_recons, :] = np.array([np.nan]*8)
            # cov_mat = np.full((8,8), np.nan)
            cpt += 1

        # cov_mats[current_recons, :, :] = cov_mat

    stds[:, 0] *= rad2deg  # std_alpha in degrees
    stds[:, 1] *= rad2deg  # std_beta in degrees
    stds[:, 4] *= rad2deg  # std_theta in degrees
    stds[:, 5] *= rad2deg  # std_phi in degrees

    if verbose and 1<2:
        print(f"Stds for first 20 coincidences:")
        for j in range(min(20, n_to_process)):
            print(f"\n\n[Coincidence {j}] \nstd_alpha={stds[j,0]:.4e}°, \nstd_beta={stds[j,1]:.4e}°, \nstd_rxmax={stds[j,2]/1e3:.4e} km, \nstd_t0={stds[j,3]:.4e}, \nstd_theta={stds[j,4]:.4e}°, \nstd_phi={stds[j,5]:.4e}°, \nstd_dw={stds[j,6]:.4e}, \nstd_Amp={stds[j,7]:.4e}")
        print(f"Percentage of singular matrices: {100.0 * cpt / n_to_process:.2f}%")
        
    print(f"\n[{time.time()-t0:.3f}s] ADF + SWF CRB done for {n_to_process} coincidences with {cpt} singular matrices\n")
    
    return stds#, cov_mats

def ADF_CRB(ncoincs: int, nants: np.ndarray, antennas_coords: np.ndarray, SWF_res: np.ndarray, ADF_res: np.ndarray, filepath: str, B_vecs: np.ndarray, n_max: int=None, verbose: bool=False) -> np.ndarray:
    """ Function calculating the Cramér-Rao Bound for the joint ADF + SWF reconstruction
    Inputs:
        ncoincs: number of coincidences
        nants: array of number of antennas per coincidence
        antennas_coords: array of antenna coordinates per coincidence
        SWF_res: dictionary containing SWF reconstruction results
        ADF_res: dictionary containing ADF reconstruction results
        file_path: path to the file containing the data
        B_vecs: array of magnetic field vectors
        n_max: maximum number of coincidences to process
        verbose: boolean for verbosity
    Outputs:
        stds: array of standard deviations for each parameter per coincidence"""

    t0           = time.time()                 # Début du timer
    n_to_process = ncoincs if n_max is None else  min(ncoincs, n_max) # Nombre de coïncidences à traiter
    stds         = np.zeros((n_to_process, 4)) # stds pré-allocation
    deg2rad      = np.pi / 180.0
    rad2deg      = 180.0 / np.pi
    cpt          = 0

    # Determine noise floor and amplitude uncertainty based on file path
    is_efield = 'efield' in filepath
    is_gp300 = 'GP300' in filepath
    is_gp289 = 'GP289' in filepath
    is_nj_adc = '-NJ_adc' in filepath
    is_an_adc = '-AN_adc' in filepath

    if is_efield:
        min_amplitude = 1e-3 # 1e-3 µV/m, minimal increment of values
        jitter_time   = 0.5e-9 # 0.5 ns, typical time resolution for electric field measurements
        galactic_noise_floor = 0.0
        amplitude_uncertainty = 0.0

    elif is_gp300:  # ZHAireS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_nj_adc:
            galactic_noise_floor = 0.0
            amplitude_uncertainty = 0.0
        elif is_an_adc:
            galactic_noise_floor = 15.0
            amplitude_uncertainty = 0.075
        else:
            galactic_noise_floor = 4.0
            amplitude_uncertainty = 0.075

    elif is_gp289:  # ZHAireS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_nj_adc:
            galactic_noise_floor = 0.0
            amplitude_uncertainty = 0.0
        elif is_an_adc:
            galactic_noise_floor = 12.0
            amplitude_uncertainty = 0.075
        else:
            galactic_noise_floor = 5.0
            amplitude_uncertainty = 0.075

    else:  # CoREAS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_an_adc:
            galactic_noise_floor = 10.0
            amplitude_uncertainty = 0.075


    for current_recons in tqdm(range(n_to_process), desc='ADF only CRB computing...'):
        n_ants = nants[current_recons]
        ant_coords = antennas_coords[current_recons, :n_ants] # Coords
        
        swf = SWF_res[current_recons]
        adf = ADF_res[current_recons]

        swf_rad = swf.copy()
        swf_rad[0] *= deg2rad
        swf_rad[1] *= deg2rad
        adf_rad = adf.copy()
        adf_rad[0] *= deg2rad
        adf_rad[1] *= deg2rad

        Xsource = cb.build_Xsource_np(swf_rad[0], swf_rad[1], swf_rad[2])
        B_vec = B_vecs[current_recons]
        fisher_mat = np.zeros((4,4))

        # Calcul des dérivées pour tous les paramètres
        h = 1e-6 * (np.abs(adf_rad)) 
        derivates_ampl = np.zeros((n_ants, 4))

        # Calcul de la dérivée pour chaque antenne, par rapport à chaque paramètre
        for i in range(4):
            # Perturbations symétriques
            params_plus  = adf_rad.copy() ; params_plus[i]  += h[i]
            params_minus = adf_rad.copy() ; params_minus[i] -= h[i]

            # Reconstruction of Xmax for perturbed SWF parameters
            
            pred_plus_ampl  = ADF_3D_model(params_plus, ant_coords, Xsource, B_vec) # in mV
            pred_minus_ampl  = ADF_3D_model(params_minus, ant_coords, Xsource, B_vec) # in mV
            
            # Dérivée
            derivates_ampl[:, i] = (pred_plus_ampl - pred_minus_ampl) / (2 * h[i])
        
        sigma_amp = amplitude_uncertainty * abs(ADF_3D_model(adf_rad, ant_coords, Xsource, B_vec))  # 7.5% amplitude uncertainty in mV
        sigma_amp = [ (sigma_amp[i]**2 + galactic_noise_floor**2 + min_amplitude**2)**0.5 for i in range(n_ants)]  # Fixed minimum amplitude uncertainty in mV
        sigma_amp = np.array(sigma_amp)
        sigma_amp = np.where(sigma_amp == 0, 1, sigma_amp) # Avoid division by zero
            
        for k in range(n_ants):
            fisher_mat += np.outer(derivates_ampl[k,:], derivates_ampl[k,:]) / (sigma_amp[k]**2)

        try:
            cov_mat = np.linalg.inv(fisher_mat)
            stds[current_recons, :] = np.sqrt(np.diag(cov_mat)) # Écarts-types
            if np.any(np.isnan(stds[current_recons, :])) or np.any(np.isinf(stds[current_recons, :])):
                if verbose and 0<1:
                    print(f"Fisher matrix inversion NaN of Inf for coinc {current_recons}.")
                    # print(fisher_mat)
                stds[current_recons, :] = np.array([np.nan]*4)
                cpt += 1
        except np.linalg.LinAlgError:
            if verbose and 0<1:
                print(f"Fisher matrix is singular for coinc {current_recons}.")
                print(fisher_mat)
            stds[current_recons, :] = np.array([np.nan]*4)
            # cov_mat = np.full((8,8), np.nan)
            cpt += 1

        # cov_mats[current_recons, :, :] = cov_mat

    stds[:, 0] *= rad2deg  # std_alpha in degrees
    stds[:, 1] *= rad2deg  # std_beta in degrees

    if verbose and 1<2:
        print(f"Stds for first 20 coincidences:")
        for j in range(min(20, n_to_process)):
            print(f"\n\n[Coincidence {j}] \nstd_theta={stds[j,0]:.4e}°, \nstd_phi={stds[j,1]:.4e}°, \nstd_dw={stds[j,2]:.4e}, \nstd_Amp={stds[j,3]:.4e}")
        print(f"Percentage of singular matrices: {100.0 * cpt / n_to_process:.2f}%")
        
    print(f"\n[{time.time()-t0:.3f}s] ADF only CRB done for {n_to_process} coincidences with {cpt} singular matrices")
    
    return stds #, cov_mats

def PWF_CRB(ncoincs: int, nants: np.ndarray, antennas_coords: np.ndarray, PWF_res: np.ndarray, file_path: str, n_max: int=None, verbose: bool=False):
    """ CRB for PWF recons """
    t0           = time.time()                 # Début du timer
    n_to_process = ncoincs if n_max is None else  min(ncoincs, n_max) # Nombre de coïncidences à traiter
    stds         = np.zeros((n_to_process, 2)) # stds pré-allocation
    deg2rad      = np.pi / 180.0
    rad2deg      = 180.0 / np.pi
    cpt          = 0

    for current_recons in tqdm(range(n_to_process), desc='PWF CRB computing...'):
        n_ants = nants[current_recons]
        ant_coords = antennas_coords[current_recons, :n_ants] # Coords
        pwf_res = PWF_res[current_recons] * deg2rad

        h = 1e-6 * (np.abs(pwf_res) + 0.1)
        derivates_time = np.zeros((n_ants, 2))
        fisher_mat = np.zeros((2,2))

        for i in range(2):
            # Perturbations symétriques
            params_plus  = pwf_res.copy() ; params_plus[i]  += h[i]
            params_minus = pwf_res.copy() ; params_minus[i] -= h[i]
            
            # Dérivée par différences finies
            pred_plus_time  = PWF_model(params_plus, ant_coords)
            pred_minus_time = PWF_model(params_minus, ant_coords)
            
            # Dérivée
            derivates_time[:, i] = (pred_plus_time - pred_minus_time) / (2 * h[i])

        sigma_time = pr.jitter_time * pr.c_light # Fixed time uncertainty in ns
        for k in range(n_ants):
            fisher_mat += np.outer(derivates_time[k,:], derivates_time[k,:]) / (sigma_time**2)

        try: 
            cov_mat = np.linalg.inv(fisher_mat)
            stds[current_recons, 0:2] = np.sqrt(np.diag(cov_mat)) # Écarts-types
            if np.any(np.isnan(stds[current_recons, 0:2])) or np.any(np.isinf(stds[current_recons, 0:2])):
                if verbose and 2<1:
                    print(f"Fisher matrix inversion NaN of Inf for coinc {current_recons}.")
                    print(fisher_mat)
                stds[current_recons, 0:2] = np.array([np.nan]*2)
                cpt += 1
        except np.linalg.LinAlgError:
            if verbose and 2<1:
                print(f"Fisher matrix is singular for coinc {current_recons}.")
                print(fisher_mat)
            stds[current_recons, 0:2] = np.array([np.nan]*2)
            cpt += 1
        
    stds[:, 0] *= rad2deg  # std_theta in degrees
    stds[:, 1] *= rad2deg  # std_phi in degrees

    if verbose and 1<2:
        print(f"Stds for first 20 coincidences:")
        for j in range(min(20, n_to_process)):
            print(f"\n\n[Coincidence {j}] \nstd_theta={stds[j,0]:.4e}°, \nstd_phi={stds[j,1]:.4e}°")
        print(f"Percentage of singular matrices: {100.0 * cpt / n_to_process:.2f}%")
    
    print(f"\n[{time.time()-t0:.3f}s] PWF CRB done for {n_to_process} coincidences with {cpt} singular matrices")

    # np.save(os.path.join(file_path, "PWF_CRB_res.npy"), 
            # {'data': stds, 'columns': ['std_theta_deg', 'std_phi_deg']}, 
            # allow_pickle=True)

def angular_error(dataframe:pd.DataFrame) -> np.ndarray:

    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi

    def std_psi(row):
        std_theta = row['stds_theta'] * deg2rad
        std_phi = row['stds_phi'] * deg2rad
        theta_recons = row['recons_theta'] * deg2rad

        std_psi = np.sqrt( (std_theta)**2 + (std_phi * np.sin(theta_recons))**2 ) * rad2deg
        return std_psi
    
    def psi(row):
        theta_recons = row['recons_theta'] * deg2rad
        phi_recons = row['recons_phi'] * deg2rad
        theta_true = row['true_theta'] * deg2rad
        phi_true = row['true_phi'] * deg2rad

        cos_psi = np.sin(theta_recons)*np.sin(theta_true)*np.cos(phi_recons - phi_true) + np.cos(theta_recons)*np.cos(theta_true)
        cos_psi = np.clip(cos_psi, -1.0, 1.0) # Clip pour éviter les erreurs numériques
        psi_rad = np.arccos(cos_psi)
        psi_deg = psi_rad * rad2deg
        return psi_deg
    
    if not dataframe.empty:
        dataframe['std_psi'] = dataframe.apply(std_psi, axis=1)
        print("\n Successfully calculated 'std_psi' for all events.")
        dataframe['psi'] = dataframe.apply(psi, axis=1)
        print("\n Successfully calculated 'psi' for all events.")

    return dataframe



# ======================= GRAMAMGE ======================= #

def grammage_reconsrtuction(SWF_res: np.ndarray, ADF_res: np.ndarray, verbose: bool=False) -> np.ndarray:

    """ Function reconstructing the grammage for all coincidences
    Inputs:
        SWF_res: array containing SWF reconstruction results (in degrees and meters)
        ADF_res: array containing ADF reconstruction results (in degrees)
    Outputs:
        grammages_g_cm2 : array of reconstructed grammages (in g/cm^2)
    """
    
    SWF_rad = SWF_res.copy()
    SWF_rad[:, :2]  *= np.pi / 180.0
    SWF_rad = jnp.array(SWF_rad)
    ADF_rad = ADF_res.copy()
    ADF_rad[:, :2] *= np.pi / 180.0
    ADF_rad = jnp.array(ADF_rad)
    grammages_g_cm2 = []
    for i in tqdm(range(SWF_rad.shape[0]), desc='Grammage reconstruction...'):
        grammage = gr.jax_slant_depth_adf_jit(SWF_rad[i, :], ADF_rad[i, :])
        grammages_g_cm2.append(grammage)
        if verbose:
            print(f"Coincidence {i}: Grammage = {grammages_g_cm2[-1]:.2f} g/cm^2")

    grammages_g_cm2 = jnp.array(grammages_g_cm2)
    return grammages_g_cm2

def Xcore_recons(SWF_res: np.ndarray, ADF_res: np.ndarray) -> np.ndarray:
    """
    Function reconstructing the core position for all coincidences
    Inputs:
        SWF_res: array containing SWF reconstruction results (in degrees and meters)
        ADF_res: array containing ADF reconstruction results (in degrees)
    Outputs:
        X_core: array of reconstructed core positions per coincidence in ENU coordinates (in meters)
    """
    
    d2r = np.pi / 180.0 # Degrees to radians conversion factor
    k_vect   = np.array(cb.build_K_vector_np(ADF_res[:,0]*d2r, ADF_res[:,1]*d2r).T) # theta and phi to have k vector
    Xsource  = np.array(cb.build_Xsource_np(SWF_res[:,0]*d2r, SWF_res[:,1]*d2r, SWF_res[:,2]).T) # Build Xsource from SWF results (alpha, beta, rxmax)
    prop_factor = np.array(pr.groundAltitude - Xsource[:,2]) / (k_vect[:,2]) # proportionnality factor to get from Xsource to Xcore (intersection with ground plane at pr.groundAltitude)

    X_core = Xsource + k_vect * prop_factor[:, np.newaxis] 
    X_core[:, 2] -= pr.groundAltitude # Convert to ground as reference (z=0 at ground level)
    return X_core



# ============================ Main ============================ #

def main():

    #  Global parameters
    if args.test:
        n_max = 20
    else:
        n_max = args.nmax if args.nmax is not None else np.inf

    file_path        = args.filepath
    multi_processing = args.multi
    verbose_bool     = False if not args.verbose else True

    # Coincidence set loading
    file_path      = args.filepath
    data_file_path = os.path.join(file_path,'data_npy')

    print(f"\n-------------- Looking for input files -----------------\n\n")
    if not os.path.exists(os.path.join(data_file_path,'co_ncoincs.npy')) or args.all or args.build : 
        print("Preprocessing input data...")
        if args.old:
            lo.old_npy_files_builder(file_path, data_file_path)
        else:
            lo.npy_files_builder(file_path, data_file_path)
        print("Input data preprocessing done.")

    print("Loading coincidence data...")
    # Load data from .npy files
    loaded_data       = lo.load_data(data_file_path, ['ncoincs', 'events_ids'])
    ncoincs           = loaded_data['ncoincs']
    events_ids_coords = loaded_data['events_ids']
    del loaded_data # Free memory

    # Limite du nombre d'événements à traiter
    n_to_process = min(ncoincs, n_max)
    events_ids_unique = events_ids_coords[:n_to_process]
    del events_ids_coords # Free memory

    # Build results dataframe
    if not os.path.exists(os.path.join(file_path, 'results_dataframe.parquet')) or args.test or args.all or args.build:
        results_df = lo.build_result_dataframe(file_path=file_path, nmax=n_to_process, old=args.old)
    else:
        results_df = pd.read_parquet(os.path.join(file_path, 'results_dataframe.parquet'))

    file_path = args.filepath if not args.test else os.path.join(args.filepath, 'CRB_test/')
    if not os.path.exists(file_path): os.makedirs(file_path)
    print(f"Loaded {n_to_process} coincidences and computing {n_to_process}.\n")

    # Looking for existing CRB, SWF, ADF, grammage and energy results to avoid recomputation if not necessary
    run_SWF = run_ADF = run_CRB = run_grammage = run_energy = True

    if 'recons_theta'    in results_df.columns : run_ADF      = False
    if 'recons_alpha'    in results_df.columns : run_SWF      = False 
    if 'stds_rxmax'      in results_df.columns : run_CRB      = False
    if 'recons_grammage' in results_df.columns : run_grammage = False
    if 'recons_energy'   in results_df.columns : run_energy   = False

    if not os.path.exists(os.path.join(file_path, 'results_dataframe.parquet')) or (time.time() - os.path.getmtime(os.path.join(file_path, 'results_dataframe.parquet'))) > 21*24*3600: 
        run_CRB = run_SWF = run_ADF = run_grammage = run_energy = True
    if args.all : run_SWF = run_ADF = run_CRB = run_grammage = run_energy = True # Force run of all steps if --all is specified

    print('-------------- Starting CRB Calculations --------------\n')

    # --- Compute PWF ---
    loaded_data = lo.load_data(data_file_path, ['nants', 'antenna_coords_array', 'peak_time_array_m'])
    nants       = loaded_data['nants']
    antenna_coords_array = loaded_data['antenna_coords_array']
    peak_time_array_m    = loaded_data['peak_time_array_m']
    del loaded_data # Free memory

    PWF_res = PWF_recons(ncoincs, nants, antenna_coords_array, peak_time_array_m, n_max=n_to_process, verbose=verbose_bool)
    print("[PWF Computed]")

    del peak_time_array_m # Free memory

    # --- Load or compute SWF ---
    peak_time_array_s = lo.load_data(data_file_path, ['peak_time_array_s'])['peak_time_array_s']
    if run_SWF or args.swf:
        print("\nComputing SWF...")
        if multi_processing:
            print(f"[MULTIPROCESSING] {n_to_process} SWF reconstruction with {max(1, min(128, mp.cpu_count() - 1))} CPUs...")
            SWF_res, SWF_losses = SWF_recons_mp(ncoincs, nants, antenna_coords_array, peak_time_array_s, PWF_res, verbose=verbose_bool, n_max=n_to_process)
        else:
            SWF_res, SWF_losses = SWF_recons(ncoincs, nants, antenna_coords_array, peak_time_array_s, PWF_res, verbose=verbose_bool, n_max=n_to_process)
        # add results to dataframe
        results_df = add_df_columns(results_df, events_ids_unique, SWF_res=SWF_res, SWF_loss=SWF_losses)
        print("[SWF computed]")
    else:
        SWF_res = results_df[['recons_alpha', 'recons_beta', 'recons_rxmax', 'recons_t0']].values
        print("[SWF loaded]")

    del peak_time_array_s # Free memory
    results_df.to_parquet(os.path.join(file_path, "results_dataframe.parquet")) # Save intermediate results
    if 'inc' in results_df.columns and 'dec' in results_df.columns:
        incs_decs = np.asarray(results_df[['inc', 'dec']].values)
        B_vecs = cb.compute_B_vec(incs_decs[:,0], incs_decs[:,1])
        print("[B_vec computed from inc/dec]")
    else:
        # If inc and dec not in dataframe, use a deafult B_vec (might ruin the ADF + CRB)
        B_vecs = np.full((n_to_process, 3), pr.B_vec_norm) 

    # --- Load or compute ADF ---
    peak_amp_array = lo.load_data(data_file_path, ['peak_amp_array'])['peak_amp_array']
    if run_ADF or args.adf:
        print("\nComputing ADF...")
        if multi_processing:
            print(f"[MULTIPROCESSING] {n_to_process} ADF reconstruction with {max(1, min(128, mp.cpu_count() - 1))} CPUs...")
            ADF_res, ADF_losses = ADF_recons_mp(ncoincs, nants, antenna_coords_array, peak_amp_array, PWF_res, SWF_res, B_vecs, verbose=verbose_bool, n_max=n_to_process)
        else:
            ADF_res, ADF_losses = ADF_recons(ncoincs, nants, antenna_coords_array, peak_amp_array, PWF_res, SWF_res, B_vecs, verbose=verbose_bool, n_max=n_to_process)
        print("[ADF computed]")
        results_df = add_df_columns(results_df, events_ids_unique, ADF_res=ADF_res, ADF_loss=ADF_losses)
    else:
        ADF_res = results_df[['recons_theta', 'recons_phi', 'recons_delta_omega', 'recons_amplitude']].values
        print("[ADF loaded]")

    del peak_amp_array # Free memory
    results_df.to_parquet(os.path.join(file_path, "results_dataframe.parquet"))

    # --- Compute CRB ---
    if run_CRB or args.crb:
        print("\nComputing CRB for ADF + SWF...")
        CRB_res = ADF_SWF_CRB(ncoincs, nants, antenna_coords_array, SWF_res, ADF_res, file_path, B_vecs, n_max=n_to_process, verbose=verbose_bool)
        CRB_ADF_only = ADF_CRB(ncoincs, nants, antenna_coords_array, SWF_res, ADF_res, file_path, B_vecs, n_max=n_to_process, verbose=verbose_bool)
        results_df = add_df_columns(results_df, events_ids_unique, CRB_res=CRB_res, CRB_ADF_only=CRB_ADF_only)
    else: 
        print("[CRB loaded]")
        CRB_res = results_df[['stds_alpha', 'stds_beta', 'stds_rxmax', 'stds_t0', 'stds_theta', 'stds_phi', 'stds_delta_omega', 'stds_amplitude']].values

    results_df.to_parquet(os.path.join(file_path, "results_dataframe.parquet"))

    results_df.to_parquet(os.path.join(file_path, "results_dataframe.parquet"))

    # --- Compute grammage estimates ---
    if run_grammage or args.grammage:
        print('\n-------------- Starting Grammage Reconstruction --------------')
        print("\nComputing grammage estimates from SWF and ADF results...")
        grammages = grammage_reconsrtuction(SWF_res, ADF_res, verbose=verbose_bool)
        results_df = add_df_columns(results_df, events_ids_unique, grammages=grammages)
    else:
        print("[Grammage NOT computed] => already in dataframe")

    results_df.to_parquet(os.path.join(file_path, "results_dataframe.parquet"))

    Xcores = Xcore_recons(SWF_res, ADF_res)
    results_df = add_df_columns(results_df, events_ids_unique, xcore=Xcores)

    results_df = angular_error(results_df)

    # Save results dataframe as .parquet
    results_df.to_parquet(os.path.join(file_path, "results_dataframe.parquet"))
    print("\nAll done.")

if __name__ == "__main__":
    main()