# Imports
import os
import time
import argparse
import numpy           as np
import params_config   as pr
import pandas          as pd
import multiprocessing as mp
from wavefronts_SEB  import *
from tqdm            import tqdm
from typing          import Tuple
from iminuit         import minimize
from scipy.optimize  import differential_evolution

# argparse setup
parser = argparse.ArgumentParser(description='Cramer-Rao Bound computation for radio-detected air showers')
parser.add_argument('--nmax'     , type=int, default=None, help='Maximum number of coincidences to process')
parser.add_argument('--filepath', type=str, default='./test_NJ/', help='Path to the input data files') # other exemple './addednoise_110uV_5antennas/'
parser.add_argument('--test'  , action='store_true', help='Run all computes in test mode with a small dataset')
parser.add_argument('--tout'  , action='store_true', help='Run all reconstructions and CRB computations')
parser.add_argument('--gen', action='store_true', help='Regenerate .npy files from text data')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output during reconstructions')
parser.add_argument('--multi', action='store_true', help='Enable multiprocessing for SWF reconstructions')
args = parser.parse_args()

# Numpy compatibility
if not hasattr(np, 'infty'):
    np.infty = np.inf


# ======================= Miscellaneous ======================== #

def npy_files_builder(file_path: str) -> None:
    """ Construction des fichiers .npy à partir des fichiers texte d'entrée
    Inputs:
        file_path: path to the input data files
    Outputs:
        Saves .npy files in the specified file_path 
    """
    print("Building .npy files from text data...")

    position_file = file_path + "/coord_antennas.txt"
    coinc_file    = file_path + "/Rec_coinctable.txt"

    # --- Antennes ---
    idx = np.loadtxt(position_file, usecols=0, dtype=int)
    coords = np.loadtxt(position_file, usecols=(1,2,3))
    init = idx.min()

    # --- Coïncidences ---
    a_i, c_i = np.loadtxt(coinc_file, usecols=(0,1), dtype=int).T
    t_s, amp = np.loadtxt(coinc_file, usecols=(2,3)).T
    t = t_s * pr.c_light

    uniq = np.unique(c_i)
    good = [u for u in uniq if np.sum(c_i==u) >= 2]
    nco = len(good)

    nants = np.array([np.sum(c_i==u) for u in good], dtype=int)  # Changé de float64 à int
    nmax = int(nants.max())

    co_ai  = np.zeros((nco, nmax), dtype=int)
    co_ac  = np.zeros((nco, nmax, 3), dtype=np.float64)
    co_ci  = np.zeros((nco, nmax), dtype=int)
    co_pt  = np.zeros((nco, nmax), dtype=np.float64)
    co_pts = np.zeros((nco, nmax), dtype=np.float64)
    co_pa  = np.zeros((nco, nmax), dtype=np.float64)

    for k, u in enumerate(good):
        m = (c_i == u)
        n = int(nants[k])  # Conversion explicite en int pour l'indexation
        co_ai[k, :n]  = a_i[m] - init
        co_ac[k, :n]  = coords[a_i[m] - init]  # Correction: utiliser l'index relatif
        co_ci[k, :n]  = c_i[m]
        co_pt[k, :n]  = t[m] - t[m].min()
        co_pts[k, :n] = t_s[m]
        co_pa[k, :n]  = amp[m]

    # --- Sauvegardes ---
    np.save(file_path+"/an_indices.npy",idx)
    np.save(file_path+"/an_coordinates.npy",coords)
    np.save(file_path+"/an_init_ant.npy",init)
    np.save(file_path+"/an_nants.npy",len(idx))

    np.save(file_path+"/co_ncoincs.npy", np.array([nco], dtype=np.float64))
    np.save(file_path+"/co_nants.npy",nants)
    np.save(file_path+"/co_nantsmax.npy",nmax)
    np.save(file_path+"/co_antenna_index_array.npy",co_ai)
    np.save(file_path+"/co_antenna_coords_array.npy",co_ac)
    np.save(file_path+"/co_coinc_index_array.npy",co_ci)
    np.save(file_path+"/co_peak_time_array.npy",co_pt) # in m
    np.save(file_path+"/co_peak_time_array_in_s.npy",co_pts) # in s
    np.save(file_path+"/co_peak_amp_array.npy",co_pa)
    pass

def build_Xsource(alpha: float, beta: float, r_xmax: float) -> np.ndarray:
    """ Build the source position vector Xsource from spherical coordinates
    Inputs:
        alpha: zenith angle in radians
        beta: azimuthal angle in radians
        r_xmax: distance to the source in meters
    Outputs:
        Xsource: source position vector in meters
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    Xsource = np.array([
        r_xmax * sa * cb,
        r_xmax * sa * sb,
        pr.groundAltitude + r_xmax * ca
    ], dtype=np.float64)

    return Xsource

# ============================ PWF ============================ #

def PWF_recons(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_time_array: np.ndarray, file_path: str, n_max: int=None, verbose: bool=False) -> np.ndarray:
    """ PWF reconstruction for all coincidences
    Inputs:
        ncoincs: number of coincidences
        nants: array of number of antennas per coincidence
        antenna_coords_array: array of antenna coordinates per coincidence
        peak_time_array: array of peak times per coincidence
    Outputs:
        PWF_res: dictionary containing PWF reconstruction results
    """

    n_to_process = ncoincs if n_max is None else min(ncoincs, n_max)

    t0 = time.time()
    rad2deg = 180.0 / np.pi
    PWF_res = np.zeros((n_to_process, 2))  # theta, phi in degrees

    for i in tqdm(range(n_to_process), desc='PWF in progress...'):
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

    print(f"\n[{time.time()-t0:.3f}s] Plane Wave Fit reconstruction done for {n_to_process} coincidences")
    np.save(os.path.join(file_path, "PWF_res.npy"), {'data': PWF_res, 'columns': ['theta_deg', 'phi_deg']}, allow_pickle=True)
    return PWF_res


# ============================ SWF ============================ #

def SWF_recons(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_time_array: np.ndarray, PWF_res: np.ndarray, file_path: str, verbose: bool=False, n_max: int=None, event_type: str='EAS'):
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
        rad2deg = 180.0 / np.pi
        deg2rad = np.pi / 180.0
        SWF_res = np.zeros((n_to_process, 4))  # alpha, beta in degrees, rxmax, t_0

        for i in tqdm(range(n_to_process), desc='SWF in progress...'):
            try:
                alpha_PWF_rad = PWF_res[i,0] * deg2rad # we use theta and phi from PWF as initial guesses for alpha and beta
                beta_PWF_rad  = PWF_res[i,1] * deg2rad # they should not be too far, but not quite exactly the same

                if event_type == 'EAS':
                    bounds = np.array([[alpha_PWF_rad - 4*deg2rad, alpha_PWF_rad + 4*deg2rad], # alpha bounds
                                    [beta_PWF_rad - 4*deg2rad, beta_PWF_rad + 4*deg2rad], # beta bounds
                                    pr.bounds[2], pr.bounds[3]], dtype=np.float64) # rxmax and t0 bounds
                else:
                    bounds = pr.bounds # wide angle bounds

                PWF_guess = np.array(bounds, dtype=np.float64).mean(axis=1)
                

                args = (antenna_coords_array[i,:nants[i]], peak_time_array[i,:nants[i]], False) # if true returns chi2/ndof
                
                resu = differential_evolution(SWF_loss, bounds, args=args, strategy='best1bin', maxiter=3000, seed=42, tol=1e-6, mutation=(0.5, 1), recombination=0.7, x0=PWF_guess)

                alpha_SWF_deg = resu.x[0] * rad2deg
                beta_SWF_deg  = (resu.x[1] % (2 * np.pi)) * rad2deg # careful with modulo 2pi
                rxmax_SWF     = resu.x[2]

                if verbose : 
                    print(f"SWF initial guess for coincidence {i}:")
                    print(f"  alpha : {PWF_guess[0]*rad2deg:10.2f}°   → {alpha_SWF_deg:10.2f}°")
                    print(f"  beta  : {PWF_guess[1]*rad2deg:10.2f}°   → {beta_SWF_deg:10.2f}°")
                    print(f"  rxmax :{PWF_guess[2]/1e3:10.2f} km  →{rxmax_SWF/1e3:10.2f} km")
                    print(f"  t0    :{PWF_guess[3]:10.2e} s   →{resu.x[3]:10.2e} s")

                SWF_res[i,0] = alpha_SWF_deg
                SWF_res[i,1] = beta_SWF_deg
                SWF_res[i,2] = rxmax_SWF

            except Exception as e:
                if verbose : print(f"SWF reconstruction failed for coincidence {i} with error: {e}")
                SWF_res[i,:] = np.nan
        
        print(f"\n[{time.time()-t0:.3f}s] Spherical Wave Fit reconstruction done for {n_to_process} coincidences")
        np.save(os.path.join(file_path, "SWF_res.npy"), {'data': SWF_res, 'columns': ['alpha_deg', 'beta_deg', 'rxmax', 't0', 'x_core', 'y_core', 'z_core']}, allow_pickle=True)
        return SWF_res

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

        args = (ant_coords, peak_time_arr, False) # if true returns chi2/ndof
        
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

        return (i, alpha_SWF_deg, beta_SWF_deg, rxmax_SWF, resu.x[3])
    except Exception as e:
        if verbose : print(f"SWF reconstruction failed for coincidence {i} with error: {e}")
        return (i, np.nan, np.nan, np.nan, np.nan)

def worker_function(args: Tuple) -> Tuple[int, float, float, float, float]:
    """Fonction wrapper pour multiprocessing (doit être picklable)"""
    i, alpha_PWF_rad, beta_PWF_rad, ant_coords, peak_time_arr, verbose, event_type = args
    return SWF_single_recon(i, alpha_PWF_rad, beta_PWF_rad, ant_coords, 
                            peak_time_arr, verbose, event_type)

def SWF_recons_mp(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_time_array: np.ndarray, PWF_res: np.ndarray, file_path: str, verbose: bool=False, event_type: str='EAS', n_max: int=None) -> np.ndarray:
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
    file_path : str - Save path
    n_max : int - Max coincidences to process
    verbose : bool - Detailed output
    event_type : str - 'EAS' or other
    
    Returns:
    --------
    SWF_res : array [n_to_process, 4] - [alpha_deg, beta_deg, rxmax, t0]
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
    for result in results:
        idx, alpha, beta, rxmax, t0 = result
        SWF_res[idx] = [alpha, beta, rxmax, t0]
    
    # Free memory
    del results
    
    print(f"\n[{time.time()-t1:.3f}s] SWF reconstruction done for {n_to_process} coincidences")
    
    # Save as .npy only
    np.save(os.path.join(file_path, "SWF_res.npy"), 
        {'data': SWF_res, 'columns': ['alpha_deg', 'beta_deg', 'rxmax', 't0']}, 
            allow_pickle=True)
    
    return SWF_res


# ============================ ADF ============================ #

def ADF_recons(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_amp_array: np.ndarray, PWF_res: np.ndarray, SWF_res: np.ndarray, file_path: str, verbose: bool=False, n_max: int=None) -> np.ndarray:
    """ ADF reconstruction for all coincidences 
    Inputs:
        ncoincs: number of coincidences
        nants: array of number of antennas per coincidence
        antenna_coords_array: array of antenna coordinates per coincidence
        peak_amp_array: array of peak amplitudes per coincidence
        PWF_res: dictionary containing PWF reconstruction results
        SWF_res: dictionary containing SWF reconstruction results

    Outputs:
        ADF_res: dictionary containing ADF reconstruction results """
    
    n_to_process = ncoincs if n_max is None else min(ncoincs, n_max)
    t0 = time.time()
    r2d, d2r = 180.0/np.pi, np.pi/180.0 # degrees to radians conversion factors
    
    # Vectorized preprocessing
    theta_PWF  = PWF_res[:n_to_process,0] * d2r
    phi_PWF    = PWF_res[:n_to_process,1] * d2r
    allpha_PWF = SWF_res[:n_to_process,0] * d2r
    beta_PWF   = SWF_res[:n_to_process,1] * d2r
    rx_max     = SWF_res[:n_to_process,2]
    ca, sa, cb, sb = np.cos(allpha_PWF), np.sin(allpha_PWF), np.cos(beta_PWF), np.sin(beta_PWF)
    
    ADF_res = np.zeros((n_to_process, 4))
    
    for i in tqdm(range(n_to_process), desc='ADF in progress...'):

        # Xmax position
        Xmax = np.array([rx_max[i]*sa[i]*cb[i], rx_max[i]*sa[i]*sb[i], pr.groundAltitude+rx_max[i]*ca[i]], dtype=np.float64)
        
        # Data slicing
        n_ants = nants[i]
        peak_amps, ant_coords = peak_amp_array[i,:n_ants], antenna_coords_array[i,:n_ants]
       
        # Bounds and initial guess
        angle_pm = 3*d2r
        bounds = np.array([[theta_PWF[i]-angle_pm, theta_PWF[i]+angle_pm],  # theta bounds
                           [phi_PWF[i]-angle_pm, phi_PWF[i]+angle_pm],      # phi bounds
                            pr.bounds[6], pr.bounds[7]], dtype=np.float64)      # dw and Amp bounds
        
        max_idx = peak_amps.argmax()
        Amp_guess = np.linalg.norm(ant_coords[max_idx]-Xmax)*peak_amps[max_idx] # propagation of 1/r to highest amplitude antenna => Amp guess = r*Amp_max
        initial_guess = np.array([theta_PWF[i], phi_PWF[i], 5, Amp_guess], dtype=np.float64)

        # Optimization
        res = minimize(ADF_loss, initial_guess, bounds=bounds, args=(peak_amps,ant_coords,Xmax, False), method='migrad', tol=1e-5)

        # Store results
        ADF_res[i] = [res.x[0]*r2d, ((res.x[1] % (2*np.pi))*r2d), res.x[2], res.x[3]] # careful with modulo 2pi
        
        if verbose :
            print(f"Xmax {i}: X={Xmax[0]:.2e}, Y={Xmax[1]:.2e}, Z={Xmax[2]-pr.groundAltitude:.2e}")
            print(f"Xmax distance to (0,0,0) : {np.linalg.norm(Xmax)/1e3:.2e} km")
            print(f"  θ  : {initial_guess[0]*r2d:10.2f}°   →   {res.x[0]*r2d:10.2f}°")
            print(f"  φ  : {initial_guess[1]*r2d:10.2f}°   →   {res.x[1]*r2d:10.2f}°")
            print(f"  dw : {initial_guess[2]:10.2f}    →   {res.x[2]:10.2f}")
            print(f"  A  : {initial_guess[3]:10.2e}    →   {res.x[3]:10.2e}")

    print(f"[{time.time()-t0:.3f}s] ADF done for {n_to_process} coincidences")
    np.save(os.path.join(file_path, "ADF_res.npy"), 
            {'data': ADF_res, 'columns': ['theta_deg','phi_deg','dw','Amp']}, allow_pickle=True)
    return ADF_res

def ADF_single_recon(i: int, theta_PWF: float, phi_PWF: float, rx_max: float, sa: float, ca: float, sb: float, cb: float, ant_coords: np.ndarray, peak_amp_arr: np.ndarray, verbose: bool=False) -> Tuple[int, float, float, float, float]:
    """Single ADF reconstruction for one coincidence, ca=cos(alpha_PWF), sa=sin(alpha_PWF), cb=cos(beta_PWF), sb=sin(beta_PWF)"""
    r2d, d2r = 180.0/np.pi, np.pi/180.0
    
    try:
        # Xmax position
        Xmax = np.array([rx_max*sa*cb, rx_max*sa*sb, pr.groundAltitude+rx_max*ca], dtype=np.float64) # ca = cos(alpha_PWF), sa = sin(alpha_PWF), cb = cos(beta_PWF), sb = sin(beta_PWF)
        
        # Bounds and initial guess
        angle_pm = 3*d2r
        bounds = np.array([[theta_PWF-angle_pm, theta_PWF+angle_pm], # theta bounds
                           [phi_PWF-angle_pm, phi_PWF+angle_pm],     # phi bounds
                           pr.bounds[6], pr.bounds[7]],  # dw and Amp bounds
                           dtype=np.float64)     

        max_idx = peak_amp_arr.argmax()
        Amp_guess = np.linalg.norm(ant_coords[max_idx]-Xmax)*peak_amp_arr[max_idx] # propagation of 1/r to highest amplitude antenna => Amp guess = r*Amp_max
        initial_guess = np.array([theta_PWF, phi_PWF, 5, Amp_guess], dtype=np.float64)
        
        # Optimization
        res = minimize(ADF_loss, initial_guess, bounds=bounds, 
                      args=(peak_amp_arr, ant_coords, Xmax, False), 
                      method='migrad', tol=1e-5)
        
        if verbose:
            print(f"Xmax {i}: X={Xmax[0]:.2e}, Y={Xmax[1]:.2e}, Z={Xmax[2]-pr.groundAltitude:.2e}")
            print(f"Xmax distance to (0,0,0) : {np.linalg.norm(Xmax)/1e3:.2e} km")
            print(f"  θ  : {initial_guess[0]*r2d:10.2f}°   →   {res.x[0]*r2d:10.2f}°")
            print(f"  φ  : {initial_guess[1]*r2d:10.2f}°   →   {res.x[1]*r2d:10.2f}°")
            print(f"  dw : {initial_guess[2]:10.2f}    →   {res.x[2]:10.2f}")
            print(f"  A  : {initial_guess[3]:10.2e}    →   {res.x[3]:10.2e}")
        
        return (i, res.x[0]*r2d, (res.x[1] % (2*np.pi))*r2d, res.x[2], res.x[3]) # careful with modulo 2pi
    
    except Exception as e:
        if verbose:
            print(f"ADF reconstruction failed for coincidence {i} with error: {e}")
            print(e)
        return (i, np.nan, np.nan, np.nan, np.nan)

def worker_function_adf(args: Tuple) -> Tuple[int, float, float, float, float]:
    """Wrapper for multiprocessing"""
    i, th, ph, rx, sa, ca, sb, cb, ant_coords, peak_amp_arr, verbose = args
    return ADF_single_recon(i, th, ph, rx, sa, ca, sb, cb, ant_coords, peak_amp_arr, verbose)

def ADF_recons_mp(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_amp_array: np.ndarray, PWF_res: np.ndarray, SWF_res: np.ndarray, file_path: str, verbose: bool=False, n_max: int=None) -> np.ndarray:
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
    file_path : str - Save path
    n_max : int - Max coincidences to process
    groundAltitude : float - Ground altitude
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
    ca, sa = np.cos(al), np.sin(al)
    cb, sb = np.cos(be), np.sin(be)
    
    # Prepare arguments (will be deleted after use)
    args_list = []
    for i in range(n_to_process):
        ant_coords = antenna_coords_array[i, :nants[i]].copy()
        peak_amp_arr = peak_amp_array[i, :nants[i]].copy()
        
        args_list.append((i, th[i], ph[i], rx[i], sa[i], ca[i], sb[i], cb[i],
                        ant_coords, peak_amp_arr, verbose))
    
    # Free memory from preprocessed arrays
    del th, ph, al, be, rx, ca, sa, cb, sb
    
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
    for result in results:
        idx, theta, phi, dw, amp = result
        ADF_res[idx] = [theta, phi, dw, amp]
    
    # Free memory
    del results
    
    # Save as .npy only
    np.save(os.path.join(file_path, "ADF_res.npy"), 
            {'data': ADF_res, 'columns': ['theta_deg', 'phi_deg', 'dw', 'Amp']}, 
            allow_pickle=True)
    
    print(f"\n[{time.time()-t0:.3f}s] ADF reconstruction done for {n_to_process} coincidences")
    return ADF_res


# ======================= Energy reconstruction ======================= #

def recons_energy_from_voltage(amplitude: float, sin_alpha: float, a=1.96e7, b=7.90e6) -> float:
    """
    Reconstructs the electromagnetic energy from the measured radio signal amplitude in ADC counts.
    Just a first proxy
    Parameters
    ----------
    amplitude : float or array-like
        Scaling factor from the best-fit ADF (in ADC units).
    sin_alpha : float or array-like
        Sine of the geomagnetic angle between the shower axis and the geomagnetic field.
    a : float
        Slope parameter obtained from the fit on simulations (default: 1.96e7).
    b : float
        Offset parameter obtained from the fit on simulations (default: 7.90e6).
    See arXiv:2507.04324 for details on how the values of a and b are determined.
    Returns
    -------
    energy : float or array-like
        Reconstructed electromagnetic energy in eV.
        Events yielding negative reconstructed energies are set to zero.
        Such cases may correspond to genuine cosmic-ray events for which the
        voltage-based energy proxy fails, as this reconstruction is not very robust
        and should only be interpreted as a first-order estimator.
    """

    energy = (amplitude / sin_alpha - b) / a   
    energy = np.maximum(energy, 0.0) * 1e18 
    return energy


# ======================= CRB of ADF + SWF ======================= #

def ADF_SWF_CRB(ncoincs: int, nants: np.ndarray, antennas_coords: np.ndarray, SWF_res: np.ndarray, ADF_res: np.ndarray, file_path: str, n_max: int=None, verbose: bool=False):

    """ Function calculating the Cramér-Rao Bound for the joint ADF + SWF reconstruction
    Inputs:
        ncoincs: number of coincidences
        nants: array of number of antennas per coincidence
        antennas_coords: array of antenna coordinates per coincidence
        SWF_res: dictionary containing SWF reconstruction results
        ADF_res: dictionary containing ADF reconstruction results
        file_path: path to save the results
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

        Xsource = build_Xsource(swf_rad[0], swf_rad[1], swf_rad[2])
        
        params = np.hstack((swf_rad, adf_rad))
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
            X_max_plus = build_Xsource(swf_params_plus[0], swf_params_plus[1], swf_params_plus[2])
            X_max_minus = build_Xsource(swf_params_minus[0], swf_params_minus[1], swf_params_minus[2])
            
            pred_plus_ampl  = ADF_3D_model(adf_params_plus, ant_coords, X_max_plus) # in mV
            pred_plus_time  = SWF_model(swf_params_plus, ant_coords) # in s

            pred_minus_ampl  = ADF_3D_model(adf_params_minus, ant_coords, X_max_minus) # in mV
            pred_minus_time  = SWF_model(swf_params_minus, ant_coords) # in s
            
            # Dérivée
            derivates_ampl[:, i] = (pred_plus_ampl - pred_minus_ampl) / (2 * h[i])
            derivates_time[:, i] = (pred_plus_time - pred_minus_time) / (2 * h[i])
        
        sigma_amp = 0.075 * abs(ADF_3D_model(adf_rad, ant_coords, Xsource))  # 7.5% amplitude uncertainty in mV
        sigma_amp = [ (sigma_amp[i]**2 + pr.galactic_noise_floor**2)**0.5 for i in range(n_ants)]  # Fixed minimum amplitude uncertainty in mV
        sigma_time = (pr.jitter_time) # Fixed time uncertainty in s
            
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
                stds[current_recons, :] = np.nan
                cpt += 1
        except np.linalg.LinAlgError:
            if verbose and 0<1:
                print(f"Fisher matrix is singular for coinc {current_recons}.")
                print(fisher_mat)
            stds[current_recons, :] = np.nan
            cpt += 1

    stds[:, 0] *= rad2deg  # std_alpha in degrees
    stds[:, 1] *= rad2deg  # std_beta in degrees
    stds[:, 4] *= rad2deg  # std_theta in degrees
    stds[:, 5] *= rad2deg  # std_phi in degrees

    if verbose and 1<2:
        print(f"Stds for first 20 coincidences:")
        for j in range(min(20, n_to_process)):
            print(f"\n\n[Coincidence {j}] \nstd_alpha={stds[j,0]:.4e}°, \nstd_beta={stds[j,1]:.4e}°, \nstd_rxmax={stds[j,2]/1e3:.4e} km, \nstd_t0={stds[j,3]:.4e}, \nstd_theta={stds[j,4]:.4e}°, \nstd_phi={stds[j,5]:.4e}°, \nstd_dw={stds[j,6]:.4e}, \nstd_Amp={stds[j,7]:.4e}")
        print(f"Percentage of singular matrices: {100.0 * cpt / n_to_process:.2f}%")
        
    print(f"\n[{time.time()-t0:.3f}s] ADF + SWF CRB done for {n_to_process} coincidences with {cpt} singular matrices")
    
    np.save(os.path.join(file_path, "CRB_res.npy"), 
            {'data': stds, 'columns': ['std_alpha_deg', 'std_beta_deg', 'std_rxmax', 'std_t0', 'std_theta_deg', 'std_phi_deg', 'std_dw', 'std_Amp']}, 
            allow_pickle=True)
    
    return stds

def PWF_CRB(ncoincs: int, nants: np.ndarray, antennas_coords: np.ndarray, PWF_res: np.ndarray, file_path: str, n_max: int=None, verbose: bool=False):
    """ CRB for PWF recons """
    t0           = time.time()                 # Début du timer
    n_to_process = ncoincs if n_max is None else  min(ncoincs, n_max) # Nombre de coïncidences à traiter
    stds         = np.zeros((n_to_process, 8)) # stds pré-allocation
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
                stds[current_recons, 0:2] = np.nan
                cpt += 1
        except np.linalg.LinAlgError:
            if verbose and 2<1:
                print(f"Fisher matrix is singular for coinc {current_recons}.")
                print(fisher_mat)
            stds[current_recons, 0:2] = np.nan
            cpt += 1
        
    stds[:, 0] *= rad2deg  # std_theta in degrees
    stds[:, 1] *= rad2deg  # std_phi in degrees

    if verbose and 1<2:
        print(f"Stds for first 20 coincidences:")
        for j in range(min(20, n_to_process)):
            print(f"\n\n[Coincidence {j}] \nstd_theta={stds[j,0]:.4e}°, \nstd_phi={stds[j,1]:.4e}°")
        print(f"Percentage of singular matrices: {100.0 * cpt / n_to_process:.2f}%")
    
    print(f"\n[{time.time()-t0:.3f}s] PWF CRB done for {n_to_process} coincidences with {cpt} singular matrices")

    np.save(os.path.join(file_path, "PWF_CRB_res.npy"), 
            {'data': stds, 'columns': ['std_theta_deg', 'std_phi_deg']}, 
            allow_pickle=True)

# ============================ Main ============================ #

def main():

    #  Global parameters
    n_max            = args.nmax if not args.test else 20
    file_path        = args.filepath
    npy_gen_bool     = args.gen
    multi_processing = args.multi
    verbose_bool     = False if not args.verbose else True

    # Coincidence set loading
    file_path              = args.filepath
    print(f"-------------- Starting CRB Calculations -----------------\nUsing data from: {file_path}")
    if not os.path.exists(os.path.join(file_path,'co_ncoincs.npy')) or npy_gen_bool == True : # If files do not exist or position file has not been modified recently
        print("Preprocessing input data...")
        npy_files_builder(file_path)
        print("Input data preprocessing done.")

    print("\nLoading coincidence data...")
    nants                =     np.load(os.path.join(file_path,'co_nants.npy'))
    antenna_coords_array =     np.load(os.path.join(file_path,'co_antenna_coords_array.npy'))
    peak_time_array_m    =     np.load(os.path.join(file_path,'co_peak_time_array.npy'))
    peak_time_array_s    =     np.load(os.path.join(file_path,'co_peak_time_array_in_s.npy'))
    peak_amp_array       =     np.load(os.path.join(file_path,'co_peak_amp_array.npy'))
    ncoincs              = int(np.load(os.path.join(file_path,'co_ncoincs.npy'))[0])
    # convert in float32 and int32 for faster processing
    antenna_coords_array = antenna_coords_array.astype(np.float64)
    peak_time_array_m     = peak_time_array_m.astype(np.float64)
    peak_time_array_s     = peak_time_array_s.astype(np.float64)
    peak_amp_array       = peak_amp_array.astype(np.float64)
    nants                = nants.astype(np.int32)
    n_to_process = min(ncoincs, n_max) if args.nmax is not None else ncoincs
    print(f"Loaded {n_to_process} coincidences.")

    file_path = args.filepath if not args.test else os.path.join(args.filepath, 'CRB_test/')
    if not os.path.exists(file_path): os.makedirs(file_path)

    files = {"PWF": "PWF_res.npy",
             "CRB": "CRB_res.npy",
             "SWF": "SWF_res.npy",
             "ADF": "ADF_res.npy"}
    
    # Check séquentiel
    run_SWF = run_ADF = True
    if os.path.exists(os.path.join(file_path, files["SWF"])):
        run_SWF = False
    if os.path.exists(os.path.join(file_path, files["ADF"])):
        run_ADF = False

    # Forcer CRB si demandé
    if args.test or args.tout : run_SWF = run_ADF = True

    # --- Load or compute PWF ---
    print("\nComputing PWF...") 
    PWF_res = PWF_recons(ncoincs, nants, antenna_coords_array, peak_time_array_m, file_path, n_max=n_max, verbose=verbose_bool)
    print("[PWF Computed]")

    # --- Load or compute SWF ---
    print("\nComputing SWF...")
    if run_SWF:
        if multi_processing:
            print(f"[MULTIPROCESSING] {n_to_process} SWF reconstruction with {mp.cpu_count()-1} CPUs...")
            SWF_res = SWF_recons_mp(ncoincs, nants, antenna_coords_array, peak_time_array_s, PWF_res, file_path, verbose=verbose_bool, n_max=n_max)
        else:
            SWF_res = SWF_recons(ncoincs, nants, antenna_coords_array, peak_time_array_s, PWF_res, file_path, verbose=verbose_bool, n_max=n_max)
    else:
        SWF_res = np.load(os.path.join(file_path, files["SWF"]), allow_pickle=True).item()['data']
        print("[SWF loaded]")

    # --- Load or compute ADF ---
    if run_ADF:
        print("\nComputing ADF...")
        if multi_processing:
            print(f"[MULTIPROCESSING] {n_to_process} ADF reconstruction with {mp.cpu_count()-1} CPUs...")
            ADF_res = ADF_recons_mp(ncoincs, nants, antenna_coords_array, peak_amp_array, PWF_res, SWF_res, file_path, verbose=verbose_bool, n_max=n_max)
        else:
            ADF_res = ADF_recons(ncoincs, nants, antenna_coords_array, peak_amp_array, PWF_res, SWF_res, file_path, verbose=verbose_bool, n_max=n_max)
    else:
        ADF_res = np.load(os.path.join(file_path, files["ADF"]), allow_pickle=True).item()['data']
        print("[ADF loaded]")

    print("\nComputing CRB for PWF...")
    PWF_CRB(ncoincs, nants, antenna_coords_array, PWF_res, file_path, n_max=n_max, verbose=verbose_bool)

    print("\nComputing CRB for ADF + SWF...")
    ADF_SWF_CRB(ncoincs, nants, antenna_coords_array, SWF_res, ADF_res, file_path, n_max=n_max, verbose=verbose_bool)


if __name__ == "__main__":
    main()