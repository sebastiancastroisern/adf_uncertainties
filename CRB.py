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
import MCEq.geometry.density_profiles as dp
import wavefronts.energy_jax          as ej
import wavefronts.params_config       as pr
import wavefronts.loader_txt          as lo
import wavefronts.Gram_Recons         as gr
from tqdm            import tqdm
from typing          import Tuple
from iminuit         import minimize
from scipy.optimize  import differential_evolution
from wavefronts.wavefronts_SEB import *

# argparse setup
parser = argparse.ArgumentParser(description='Cramer-Rao Bound computation for radio-detected air showers')
parser.add_argument('--nmax'    ,type=int, default=None, help='Maximum number of coincidences to process')
parser.add_argument('--filepath', type=str, default='./test_NJ/', help='Path to the input data files') # other exemple './addednoise_110uV_5antennas/'
parser.add_argument('--test'   , action='store_true', help='Run all computes in test mode with a small dataset')
parser.add_argument('--tout'   , action='store_true', help='Run all reconstructions and CRB computations')
parser.add_argument('--gen'    , action='store_true', help='Regenerate .npy files from text data')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output during reconstructions')
parser.add_argument('--multi'  , action='store_false', help='Enable multiprocessing for SWF reconstructions. Default is True (multiprocessing enabled)')
parser.add_argument('--savemat', action='store_false', help='Save Fisher information matrices as .npy files')
args = parser.parse_args()

# Numpy compatibility
if not hasattr(np, 'infty'):
    np.infty = np.inf


# ======================= Miscellaneous ======================== #

def npy_files_builder(file_path: str, data_filepath: str) -> None:
    """ Construction des fichiers .npy à partir des fichiers texte d'entrée
    Inputs:
        file_path: path to the input data files
        data_filepath: path to the data files (e.g., './test_NJ/')
    Outputs:
        Saves .npy files in the specified file_path 
    """
    print("Building .npy files from text data...")

    if not os.path.exists(data_filepath): os.makedirs(data_filepath)
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
    np.save(data_filepath+"/an_indices.npy",idx)
    np.save(data_filepath+"/an_coordinates.npy",coords)
    np.save(data_filepath+"/an_init_ant.npy",init)
    np.save(data_filepath+"/an_nants.npy",len(idx))

    np.save(data_filepath+"/co_ncoincs.npy", np.array([nco], dtype=np.float64))
    np.save(data_filepath+"/co_nants.npy",nants)
    np.save(data_filepath+"/co_nantsmax.npy",nmax)
    np.save(data_filepath+"/co_antenna_index_array.npy",co_ai)
    np.save(data_filepath+"/co_antenna_coords_array.npy",co_ac)
    np.save(data_filepath+"/co_coinc_index_array.npy",co_ci)
    np.save(data_filepath+"/co_peak_time_array.npy",co_pt) # in m
    np.save(data_filepath+"/co_peak_time_array_in_s.npy",co_pts) # in s
    np.save(data_filepath+"/co_peak_amp_array.npy",co_pa)
    pass

def build_Xsource(alpha_rad: float, beta_rad: float, r_xmax: float) -> np.ndarray:
    """ Build the source position vector Xsource from spherical coordinates
    Inputs:
        alpha_rad: zenith angle in radians
        beta_rad: azimuthal angle in radians
        r_xmax: distance to the source in meters
    Outputs:
        Xsource: source position vector in meters
    """
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    cb = np.cos(beta_rad)
    sb = np.sin(beta_rad)

    Xsource = np.array([
        r_xmax * sa * cb,
        r_xmax * sa * sb,
        pr.groundAltitude + r_xmax * ca
    ], dtype=np.float64)

    return Xsource

def build_K_vector(theta: float, phi: float) -> np.ndarray:
    """ Build the shower direction vector K from spherical coordinates
    Inputs:
        theta: zenith angle in radians
        phi: azimuthal angle in radians
    Outputs:
        K: shower direction vector
    """
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)

    K = np.array([
        -st * cp,
        -st * sp,
        -ct
    ], dtype=np.float64)

    return K

def build_result_dataframe(file_path: str= args.filepath, nmax: int=None) -> pd.DataFrame:
    """ Build a pandas DataFrame from the reconstruction results and CRB computations
    Inputs:
        PWF_res: array containing PWF reconstruction results
        SWF_res: array containing SWF reconstruction results
        ADF_res: array containing ADF reconstruction results
        CRB_res: array containing CRB results
        cov_mats: array containing Fisher information matrices
    Outputs:        
        df: pandas DataFrame containing all results
    """

    if file_path.endswith('AN3-08_dec_25') or file_path.endswith('test_AN3') or file_path.endswith('test_NJ') or file_path.endswith('DC2_Training_L1'):
        df_temp = pd.read_csv(os.path.join(file_path, "input_simus.txt"), comment="#", sep=r'\s+', header=None, usecols=[1, 2, 3, 5, 6, 7, 8, 9, 10, 15], 
                        names=['true_theta', 'true_phi', 'Primary_energy', 'Nature_primary', 'XmaxDistance', 'gramage', 'x_Xmax', 'y_Xmax', 'z_Xmax', 'Number_triggered_antennas'])
    else:
        df_temp = pd.read_csv(os.path.join(file_path, "input_simus.txt"), comment="#", sep=r'\s+', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
                        names=['true_theta', 'true_phi', 'Primary_energy', 'Em_energy', 'Nature_primary', 'XmaxDistance', 'gramage', 'x_Xmax', 'y_Xmax', 'z_Xmax', 'Number_triggered_antennas'])
    
    if nmax is not None:
        df_temp = df_temp.iloc[:nmax]

    return df_temp

def add_df_columns(df: pd.DataFrame, SWF_res: np.ndarray=None, SWF_loss: np.ndarray=None, ADF_res: np.ndarray=None, ADF_loss: np.ndarray=None, CRB_res: np.ndarray=None, energies: np.ndarray=None, energies_uncertainty: np.ndarray=None, grammages: np.ndarray=None) -> pd.DataFrame:
    """ Add columns to the DataFrame with reconstruction results
    Inputs:
        df: pandas DataFrame
        SWF_res: array containing SWF reconstruction results
        SWF_loss: array containing SWF loss values
        ADF_res: array containing ADF reconstruction results
        ADF_loss: array containing ADF loss values
        CRB_res: array containing CRB results
        energies: array containing reconstructed energies
        energies_uncertainty: array containing uncertainties on reconstructed energies
        grammages: array containing reconstructed grammages
    Outputs:
        df: pandas DataFrame with added columns
    """
    if SWF_res is not None:
        new_cols = np.array([[SWF_res[i][0], SWF_res[i][1], SWF_res[i][2], SWF_res[i][3]] for i in range(len(SWF_res))])
        df[['recons_alpha', 'recons_beta', 'recons_rxmax', 'recons_t0']] = new_cols

    if SWF_loss is not None:
        df['SWF_loss'] = SWF_loss

    if ADF_res is not None:
        new_cols = np.array([[ADF_res[i][0], ADF_res[i][1], ADF_res[i][2], ADF_res[i][3]] for i in range(len(ADF_res))])
        df[['recons_theta', 'recons_phi', 'recons_delta_omega', 'recons_amplitude']] = new_cols

    if ADF_loss is not None:
        df['ADF_loss'] = ADF_loss

    if CRB_res is not None:
        new_cols = np.array([[CRB_res[i][0], CRB_res[i][1], CRB_res[i][2], CRB_res[i][3], CRB_res[i][4], CRB_res[i][5], CRB_res[i][6], CRB_res[i][7]] for i in range(len(CRB_res))])
        df[['stds_alpha', 'stds_beta', 'stds_rxmax', 'stds_t0', 'stds_theta', 'stds_phi', 'stds_delta_omega', 'stds_amplitude']] = new_cols

    if energies is not None and energies_uncertainty is not None:
        new_cols = np.array([[energies[i], energies_uncertainty[i]] for i in range(len(energies))])
        df[['recons_energy', 'recons_energy_uncertainty']] = new_cols

    if grammages is not None:
        df['recons_grammage'] = grammages

    return df

# ============================ PWF ============================ #

def PWF_recons(ncoincs: int, nants: np.ndarray, antenna_coords_array: np.ndarray, peak_time_array: np.ndarray, n_max: int=None, verbose: bool=False) -> np.ndarray:
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

    print(f"[{time.time()-t0:.3f}s] Plane Wave Fit reconstruction done for {n_to_process} coincidences")
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
        # np.save(os.path.join(file_path, "SWF_res.npy"), {'data': SWF_res, 'columns': ['alpha_deg', 'beta_deg', 'rxmax', 't0'], 'loss': SWF_losses})

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
    
    # Save as .npy only
    # np.save(os.path.join(file_path, "SWF_res.npy"), 
    #         {'data': SWF_res, 'columns': ['alpha_deg', 'beta_deg', 'rxmax', 't0'], 'loss': SWF_losses})
    
    return SWF_res, SWF_losses


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
    theta_PWF_rad = PWF_res[:n_to_process,0] * d2r
    phi_PWF_rad   = PWF_res[:n_to_process,1] * d2r
    alpha_PWF_rad = SWF_res[:n_to_process,0] * d2r
    beta_PWF_rad  = SWF_res[:n_to_process,1] * d2r
    rx_max        = SWF_res[:n_to_process,2]
    
    ADF_res = np.zeros((n_to_process, 4))
    ADF_losses = np.zeros(n_to_process)
    
    for i in tqdm(range(n_to_process), desc='ADF in progress...'):

        i, theta_deg, phi_deg, dw, Amp, loss = ADF_single_recon(i, theta_PWF_rad[i], phi_PWF_rad[i], rx_max[i], alpha_PWF_rad[i], beta_PWF_rad[i], antenna_coords_array[i,:nants[i]], peak_amp_array[i,:nants[i]], verbose)

        ADF_res[i,0] = theta_deg
        ADF_res[i,1] = phi_deg
        ADF_res[i,2] = dw
        ADF_res[i,3] = Amp
        ADF_losses[i] = loss

    print(f"[{time.time()-t0:.3f}s] ADF done for {n_to_process} coincidences")
    # np.save(os.path.join(file_path, "ADF_res.npy"), 
            # {'data': ADF_res, 'columns': ['theta_deg','phi_deg','dw','Amp'], 'loss': ADF_losses})
    return ADF_res, ADF_losses

def ADF_single_recon(i: int, theta_PWF_rad: float, phi_PWF_rad: float, rx_max: float, alpha_rad: float, beta_rad: float, ant_coords: np.ndarray, peak_amp_arr: np.ndarray, verbose: bool=False) -> Tuple[int, float, float, float, float]:
    """Single ADF reconstruction for one coincidence, ca=cos(alpha_PWF), sa=sin(alpha_PWF), cb=cos(beta_PWF), sb=sin(beta_PWF)"""
    r2d, d2r = 180.0/np.pi, np.pi/180.0
    
    try:
        # Xmax position
        Xmax = build_Xsource(alpha_rad, beta_rad, rx_max)

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
                      args=(peak_amp_arr, ant_coords, Xmax, False), 
                      method='migrad', tol=1e-5)
        
        if verbose:
            print(f"Xmax {i}: X={Xmax[0]:.2e}, Y={Xmax[1]:.2e}, Z={Xmax[2]-pr.groundAltitude:.2e}")
            print(f"Xmax distance to (0,0,0) : {np.linalg.norm(Xmax)/1e3:.2e} km")
            print(f"  θ  : {initial_guess[0]*r2d:10.2f}°   →   {res.x[0]*r2d:10.2f}°")
            print(f"  φ  : {initial_guess[1]*r2d:10.2f}°   →   {res.x[1]*r2d:10.2f}°")
            print(f"  dw : {initial_guess[2]:10.2f}    →   {res.x[2]:10.2f}")
            print(f"  A  : {initial_guess[3]:10.2e}    →   {res.x[3]:10.2e}")
        
        return (i, res.x[0]*r2d, (res.x[1] % (2*np.pi))*r2d, res.x[2], res.x[3], res.fun) # careful with modulo 2pi
    
    except Exception as e:
        if verbose:
            print(f"ADF reconstruction failed for coincidence {i} with error: {e}")
            print(e)
        return (i, np.nan, np.nan, np.nan, np.nan, np.nan)

def worker_function_adf(args: Tuple) -> Tuple[int, float, float, float, float]:
    """Wrapper for multiprocessing"""
    i, th, ph, rx, al, be, ant_coords, peak_amp_arr, verbose = args
    return ADF_single_recon(i, th, ph, rx, al, be, ant_coords, peak_amp_arr, verbose)

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
    
    # Prepare arguments (will be deleted after use)
    args_list = []
    for i in range(n_to_process):
        ant_coords = antenna_coords_array[i, :nants[i]].copy()
        peak_amp_arr = peak_amp_array[i, :nants[i]].copy()
        
        args_list.append((i, th[i], ph[i], rx[i], al[i], be[i],
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
    
    # Save as .npy only
    # np.save(os.path.join(file_path, "ADF_res.npy"), 
            # {'data': ADF_res, 'columns': ['theta_deg', 'phi_deg', 'dw', 'Amp'], 'loss': ADF_losses})
    
    print(f"\n[{time.time()-t0:.3f}s] ADF reconstruction done for {n_to_process} coincidences")
    return ADF_res, ADF_losses


# ======================= Energy reconstruction ======================= #

def recons_energy_all_cov(ncoincs: int, ADF_deg: np.ndarray, SWF_deg: np.ndarray, cov_mats: np.ndarray, output_path: str, csv_file_path: str=pr.csv_coeff_corr, verbose: bool=False, n_max: int=None) -> np.ndarray:
    """ Function reconstructing the energy for all coincidences
    Inputs:
        ncoincs: number of coincidences
        ADF_res: dictionary containing ADF reconstruction results (in degrees)
        SWF_res: dictionary containing SWF reconstruction results (in degrees)
        file_path: path to save the results
        n_max: maximum number of coincidences to process
        verbose: boolean for verbosity
        csv_file_path: path to the CSV file containing the correction coefficients
    Outputs:
        energies: array of reconstructed energies per coincidence in eV
        energies_uncertainty: array of uncertainties of reconstructed energies per coincidence in eV """
    
    if n_max is not None:
        ncoincs = min(ncoincs, n_max)

    deg2rad = np.pi / 180.0
    ADF_rad, SWF_rad = ADF_deg.copy(), SWF_deg.copy()
    ADF_rad[:,:2] *= deg2rad
    SWF_rad[:,:2] *= deg2rad
    params_rad = np.hstack((SWF_rad, ADF_rad)) # shape (ncoincs, 8)
    params_rad = jnp.array(params_rad)
    cov_mats   = jnp.array(cov_mats)

    df_coeffs = pd.read_csv(csv_file_path)
    df_coeffs = df_coeffs[df_coeffs['value'] != 0.0]
    jpn_coeffs = jnp.array(df_coeffs['value'].values)

    energies = np.zeros(ncoincs)
    energies_uncertainty = np.zeros(ncoincs)

    for i in tqdm(range(ncoincs), desc='Energy reconstruction...'):
        SWF_ADF_rad = params_rad[i]
        cov_mat     = cov_mats[i]
        energies[i], energies_uncertainty[i] = jax.jit(ej.jax_energy_and_uncertainty)(SWF_ADF_rad, cov_mat, jpn_coeffs)
        if verbose:
            print(f"Coincidence {i}: Energy = {energies[i]:.3e} EeV ± {energies_uncertainty[i]:.3e} EeV")
    
    # np.save(os.path.join(output_path, "energies.npy"), {'energies': energies*1e18, 'uncertainties': energies_uncertainty*1e18, 'columns': ['energy_eV', 'energy_uncertainty_eV']}, allow_pickle=True)

    return energies, energies_uncertainty

def recons_energy_all_crb(ncoincs: int, ADF_deg: np.ndarray, SWF_deg: np.ndarray, CRB_res: np.ndarray, output_path: str, csv_file_path: str=pr.csv_coeff_corr, verbose: bool=False, n_max: int=None) -> np.ndarray:
    """ Function reconstructing the energy for all coincidences
    Inputs:
        ncoincs: number of coincidences
        ADF_res: dictionary containing ADF reconstruction results (in degrees)
        SWF_res: dictionary containing SWF reconstruction results (in degrees)
        file_path: path to save the results
        n_max: maximum number of coincidences to process
        verbose: boolean for verbosity
        csv_file_path: path to the CSV file containing the correction coefficients
    Outputs:
        energies: array of reconstructed energies per coincidence in eV
        energies_uncertainty: array of uncertainties of reconstructed energies per coincidence in eV """
    
    t0 = time.time()
    
    if n_max is not None:
        ncoincs = min(ncoincs, n_max)

    deg2rad = np.pi / 180.0
    ADF_rad, SWF_rad, CRB_rad = ADF_deg.copy(), SWF_deg.copy(), CRB_res.copy()
    ADF_rad[:, :2] *= deg2rad
    SWF_rad[:, :2] *= deg2rad
    CRB_rad[:, :2] *= deg2rad
    CRB_rad[:,4:6] *= deg2rad
    params_rad = np.hstack((SWF_rad, ADF_rad)) # shape (ncoincs, 8)
    params_rad = jnp.array(params_rad)
    cov_mats = []
    for i in range(ncoincs):
        cov_mats.append(np.diag(CRB_rad[i]**2))
    cov_mats   = jnp.array(cov_mats)

    df_coeffs = pd.read_csv(csv_file_path)
    df_coeffs = df_coeffs[df_coeffs['value'] != 0.0]
    jpn_coeffs = jnp.array(df_coeffs['value'].values)

    energies = np.zeros(ncoincs)
    energies_uncertainty = np.zeros(ncoincs)

    for i in tqdm(range(ncoincs), desc='Energy reconstruction...'):
        SWF_ADF_rad = params_rad[i]
        cov_mat     = cov_mats[i]
        energies[i], energies_uncertainty[i] = jax.jit(ej.jax_energy_and_uncertainty)(SWF_ADF_rad, cov_mat, jpn_coeffs)
        if verbose:
            print(f"Coincidence {i}: Energy = {energies[i]:.3e} EeV ± {energies_uncertainty[i]:.3e} EeV")

    print(f"\n[{time.time()-t0:.3f}s] Energy reconstruction done for {ncoincs} coincidences")
    print(f"Percentages of negative energies: {(energies < 0).sum() / ncoincs * 100:.2f}%")
    
    # np.save(os.path.join(output_path, "energies.npy"), {'energies': energies*1e18, 'uncertainties': energies_uncertainty*1e18, 'columns': ['energy_eV', 'energy_uncertainty_eV']}, allow_pickle=True)

    return energies, energies_uncertainty


# ======================= CRB of ADF + SWF ======================= #

def ADF_SWF_CRB(ncoincs: int, nants: np.ndarray, antennas_coords: np.ndarray, SWF_res: np.ndarray, ADF_res: np.ndarray, file_path: str, n_max: int=None, verbose: bool=False, save_mat:bool=False) -> np.ndarray:

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
    cov_mats     = np.full((n_to_process, 8, 8), np.nan)

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
        
        sigma_amp = pr.amplitude_uncertainty * abs(ADF_3D_model(adf_rad, ant_coords, Xsource))  # 7.5% amplitude uncertainty in mV
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
                stds[current_recons, :] = np.array([np.nan]*8)
                cpt += 1
        except np.linalg.LinAlgError:
            if verbose and 0<1:
                print(f"Fisher matrix is singular for coinc {current_recons}.")
                print(fisher_mat)
            stds[current_recons, :] = np.array([np.nan]*8)
            cov_mat = np.full((8,8), np.nan)
            cpt += 1

        cov_mats[current_recons, :, :] = cov_mat

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
    
    # np.save(os.path.join(file_path, "CRB_res.npy"), 
    #         {'data': stds, 'columns': ['std_alpha_deg', 'std_beta_deg', 'std_rxmax', 'std_t0', 'std_theta_deg', 'std_phi_deg', 'std_dw', 'std_Amp']}, 
    #         allow_pickle=True)
    
    # if save_mat:
    #     np.save(os.path.join(file_path, "CRB_fisher_matrices.npy"), 
    #             {'data': cov_mats}, 
    #             allow_pickle=True)
    return stds, cov_mats

def ADF_CRB(ncoincs: int, nants: np.ndarray, antennas_coords: np.ndarray, SWF_res: np.ndarray, ADF_res: np.ndarray, file_path: str, n_max: int=None, verbose: bool=False, save_mat:bool=False) -> np.ndarray:
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
    stds         = np.zeros((n_to_process, 4)) # stds pré-allocation
    deg2rad      = np.pi / 180.0
    rad2deg      = 180.0 / np.pi
    cpt          = 0
    cov_mats     = np.full((n_to_process, 4, 4), np.nan)

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
            
            pred_plus_ampl  = ADF_3D_model(params_plus, ant_coords, Xsource) # in mV
            pred_minus_ampl  = ADF_3D_model(params_minus, ant_coords, Xsource) # in mV
            
            # Dérivée
            derivates_ampl[:, i] = (pred_plus_ampl - pred_minus_ampl) / (2 * h[i])
        
        sigma_amp = pr.amplitude_uncertainty * abs(ADF_3D_model(adf_rad, ant_coords, Xsource))  # 7.5% amplitude uncertainty in mV
        sigma_amp = [ (sigma_amp[i]**2 + pr.galactic_noise_floor**2)**0.5 for i in range(n_ants)]  # Fixed minimum amplitude uncertainty in mV
            
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
            cov_mat = np.full((8,8), np.nan)
            cpt += 1

        cov_mats[current_recons, :, :] = cov_mat

    stds[:, 0] *= rad2deg  # std_alpha in degrees
    stds[:, 1] *= rad2deg  # std_beta in degrees

    if verbose and 1<2:
        print(f"Stds for first 20 coincidences:")
        for j in range(min(20, n_to_process)):
            print(f"\n\n[Coincidence {j}] \nstd_theta={stds[j,0]:.4e}°, \nstd_phi={stds[j,1]:.4e}°, \nstd_dw={stds[j,2]:.4e}, \nstd_Amp={stds[j,3]:.4e}")
        print(f"Percentage of singular matrices: {100.0 * cpt / n_to_process:.2f}%")
        
    print(f"\n[{time.time()-t0:.3f}s] ADF only CRB done for {n_to_process} coincidences with {cpt} singular matrices")
    
    np.save(os.path.join(file_path, "CRB_ADF_only_res.npy"), 
            {'data': stds, 'columns': ['std_theta_deg', 'std_phi_deg', 'std_dw', 'std_Amp']}, 
            allow_pickle=True)
    
    # if save_mat:
    #     np.save(os.path.join(file_path, "CRB_ADF_only_fisher_matrices.npy"), 
    #             {'data': cov_mats}, 
    #             allow_pickle=True)
    return stds, cov_mats

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

# ======================= GRAMAMGE ======================= #

def grammage_reconsrtuction(df_results: pd.DataFrame) -> pd.DataFrame:

    """ Function reconstructing the grammage for all coincidences
    Inputs:
        df_results: dataframe containing the reconstruction results (in degrees)
    Outputs:
        df_results: dataframe with added grammage column (in g/cm^2) """
    
    SWF_deg = df_results[["recons_alpha", "recons_beta", "recons_rxmax", "recons_t0"]].values
    SWF_rad = SWF_deg.copy()
    SWF_rad[:, :2] *= np.pi / 180.0
    SWF_deg, SWF_rad = np.array(SWF_deg), np.array(SWF_rad)
    std_atm = dp.CorsikaAtmosphere('USStd')

    Xmax         = gr.compute_Xmax(SWF_rad)
    Xmax_heights = gr.conversion_to_enu_numpy(Xmax)
    grammages    = gr.compute_grammage_numpy(Xmax_heights, SWF_deg, std_atm)

    return grammages



# ============================ Main ============================ #

def main():

    #  Global parameters
    n_max            = args.nmax if not args.test else 20
    file_path        = args.filepath
    npy_gen_bool     = args.gen
    multi_processing = args.multi
    verbose_bool     = False if not args.verbose else True

    # Coincidence set loading
    file_path      = args.filepath
    data_file_path = os.path.join(file_path,'data_npy')

    print(f"-------------- Looking for input files -----------------\nUsing data from: {file_path}")
    if not os.path.exists(os.path.join(data_file_path,'co_ncoincs.npy')) or npy_gen_bool == True : # If files do not exist or position file has not been modified recently
        print("Preprocessing input data...")
        npy_files_builder(file_path, data_file_path)
        print("Input data preprocessing done.")

    print("Loading coincidence data...")
    # Load data from .npy files
    nants, antenna_coords_array, peak_time_array_m, peak_time_array_s, peak_amp_array, ncoincs = lo.load_data_files_np(file_path)
    n_to_process = min(ncoincs, n_max) if args.nmax is not None else ncoincs

    # Build results dataframe
    if not os.path.exists(os.path.join(file_path, 'results_dataframe.parquet')) or args.test:
        results_df = build_result_dataframe(file_path=file_path, nmax = n_max)
    else:
        results_df = pd.read_parquet(os.path.join(file_path, 'results_dataframe.parquet'))
    
    file_path = args.filepath if not args.test else os.path.join(args.filepath, 'CRB_test/')
    if not os.path.exists(file_path): os.makedirs(file_path)
    print(f"Loaded {n_to_process} coincidences but computing only {n_to_process}.\n")

    # Looking for existing CRB, SWF and ADF results to avoid recomputation if not necessary
    run_SWF = run_ADF = run_CRB = True
    if 'recons_theta' in results_df.columns : run_ADF = False
    if 'recons_alpha' in results_df.columns : run_SWF = False 
    if 'stds_rxmax'   in results_df.columns : run_CRB = False
    if not os.path.exists(os.path.join(file_path, 'results_dataframe.parquet')): # or (os.path.getmtime(os.path.join(file_path, 'results_dataframe.parquet')) - time.time()) > 7*24*3600: 
        run_CRB = run_SWF = run_ADF = True
    if args.tout : run_SWF = run_ADF = run_CRB = True # Forcer CRB si demandé
    
    print('-------------- Starting CRB Calculations --------------')

    # --- Compute PWF ---
    PWF_res = PWF_recons(ncoincs, nants, antenna_coords_array, peak_time_array_m, n_max=n_max, verbose=verbose_bool)
    print("[PWF Computed]")

    # --- Load or compute SWF ---
    if run_SWF:
        print("\nComputing SWF...")
        if multi_processing:
            print(f"[MULTIPROCESSING] {n_to_process} SWF reconstruction with {mp.cpu_count()-1} CPUs...")
            SWF_res, SWF_losses = SWF_recons_mp(ncoincs, nants, antenna_coords_array, peak_time_array_s, PWF_res, file_path, verbose=verbose_bool, n_max=n_max)
        else:
            SWF_res, SWF_losses = SWF_recons(ncoincs, nants, antenna_coords_array, peak_time_array_s, PWF_res, file_path, verbose=verbose_bool, n_max=n_max)
        # add results to dataframe
        results_df = add_df_columns(results_df, SWF_res=SWF_res, SWF_loss=SWF_losses)
        print("[SWF computed]")
    else:
        SWF_res = results_df[['recons_alpha', 'recons_beta', 'recons_rxmax', 'recons_t0']].values
        print("[SWF loaded]")


    # --- Load or compute ADF ---
    if run_ADF:
        print("\nComputing ADF...")
        if multi_processing:
            print(f"[MULTIPROCESSING] {n_to_process} ADF reconstruction with {mp.cpu_count()-1} CPUs...")
            ADF_res, ADF_losses = ADF_recons_mp(ncoincs, nants, antenna_coords_array, peak_amp_array, PWF_res, SWF_res, file_path, verbose=verbose_bool, n_max=n_max)
        else:
            ADF_res, ADF_losses = ADF_recons(ncoincs, nants, antenna_coords_array, peak_amp_array, PWF_res, SWF_res, file_path, verbose=verbose_bool, n_max=n_max)
        print("[ADF computed]")
        results_df = add_df_columns(results_df, ADF_res=ADF_res, ADF_loss=ADF_losses)
    else:
        ADF_res = results_df[['recons_theta', 'recons_phi', 'recons_delta_omega', 'recons_amplitude']].values
        print("[ADF loaded]")


    # --- Compute CRB ---

    if run_CRB:
        print("\nComputing CRB for ADF + SWF...")
        CRB_res, cov_mats = ADF_SWF_CRB(ncoincs, nants, antenna_coords_array, SWF_res, ADF_res, file_path, n_max=n_max, verbose=verbose_bool, save_mat=args.savemat)

        CRB_ADF_only, cov_mats_ADF_only = ADF_CRB(ncoincs, nants, antenna_coords_array, SWF_res, ADF_res, file_path, n_max=n_max, verbose=verbose_bool, save_mat=args.savemat)
        results_df = add_df_columns(results_df, CRB_res=CRB_res)
    
    else: 
        CRB_res = results_df[['stds_alpha', 'stds_beta', 'stds_rxmax', 'stds_t0', 'stds_theta', 'stds_phi', 'stds_delta_omega', 'stds_amplitude']].values


    print('\n-------------- Starting Energy Reconstruction --------------')

    # --- Compute energy estimates ---
    print("\nComputing energy estimates from ADF results...")

    energies, energies_uncertainty = recons_energy_all_crb(ncoincs, ADF_res, SWF_res, CRB_res, file_path, csv_file_path=pr.csv_coeff_corr, verbose=verbose_bool, n_max=n_max)

    results_df = add_df_columns(results_df, energies=energies, energies_uncertainty=energies_uncertainty)
    results_df.to_parquet(os.path.join(file_path, "results_dataframe.parquet"))

    print('\n-------------- Starting Grammage Reconstruction --------------')

    # --- Compute grammage estimates ---
    print("\nComputing grammage estimates from SWF results...")
    grammages = grammage_reconsrtuction(results_df)
    results_df = add_df_columns(results_df, grammages=grammages)


    print("\nAll done.")

if __name__ == "__main__":
    main()