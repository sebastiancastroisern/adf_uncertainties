import os
import jax.numpy as jnp
import numpy as np
import pandas as pd
import wavefronts.params_config as pr
import wavefronts.energy_jax as nrjax
import wavefronts.compute_basic as cmpt

LOADERS_NP = {
    'nants': lambda file_path: np.load(os.path.join(file_path,'co_nants.npy')).astype(int),
    'antenna_coords_array': lambda file_path: np.load(os.path.join(file_path,'co_antenna_coords_array.npy')).astype(float),
    'peak_time_array_m': lambda file_path: np.load(os.path.join(file_path,'co_peak_time_array.npy')).astype(float),
    'peak_time_array_s': lambda file_path: np.load(os.path.join(file_path,'co_peak_time_array_in_s.npy')).astype(float),
    'peak_amp_array': lambda file_path: np.load(os.path.join(file_path,'co_peak_amp_array.npy')).astype(float),
    'ncoincs': lambda file_path: int(np.load(os.path.join(file_path,'co_ncoincs.npy'))[0]),
    'events_ids': lambda file_path: np.load(os.path.join(file_path,'an_event_indices.npy')).astype(int)
}

LOADERS_JNP = {
    'nants': lambda file_path: jnp.load(os.path.join(file_path,'co_nants.npy')).astype(int),
    'antenna_coords_array': lambda file_path: jnp.load(os.path.join(file_path,'co_antenna_coords_array.npy')).astype(float),
    'peak_time_array_m': lambda file_path: jnp.load(os.path.join(file_path,'co_peak_time_array.npy')).astype(float),
    'peak_time_array_s': lambda file_path: jnp.load(os.path.join(file_path,'co_peak_time_array_in_s.npy')).astype(float),
    'peak_amp_array': lambda file_path: jnp.load(os.path.join(file_path,'co_peak_amp_array.npy')).astype(float),
    'ncoincs': lambda file_path: int(jnp.load(os.path.join(file_path,'co_ncoincs.npy'))[0]),
    'events_ids': lambda file_path: jnp.load(os.path.join(file_path,'an_event_indices.npy')).astype(int)
}

def load_data(data_file_path: str, needed_keys: list, use_jnp: bool=False) -> dict:
    """ Load data files from numpy files using either JAX or NumPy.
    Inputs:
        data_file_path: str
            Path to the directory containing the result files
        needed_keys: list
            List of keys corresponding to the data to load (e.g., 'nants', 'antenna_coords_array', etc.)
        use_jnp: bool, optional
            Whether to use JAX (True) or NumPy (False) for loading the data (default is False)
            Outputs:   
    """
    pass
    ressources = {}
    for name in needed_keys:
        if use_jnp:
            if name in LOADERS_JNP:
                ressources[name] = LOADERS_JNP[name](data_file_path)
            else:
                raise ValueError(f"Key '{name}' not found in JAX loaders.")
        else:
            if name in LOADERS_NP:
                ressources[name] = LOADERS_NP[name](data_file_path)
            else:
                raise ValueError(f"Key '{name}' not found in NumPy loaders.")

    return ressources

def build_result_dataframe(file_path:str, nmax:int =None, old:bool =False) -> pd.DataFrame:
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

    if old:
        df_temp = pd.read_csv(os.path.join(file_path, "input_simus.txt"), comment="#", sep=r'\s+', header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
                        names=['event_idx', 'true_theta', 'true_phi', 'Primary_energy', 'Em_energy', 'Nature_primary', 'XmaxDistance', 'gramage', 'x_Xmax', 'y_Xmax', 'z_Xmax', 'Number_triggered_antennas'])
    else:
        df_temp = pd.read_csv(os.path.join(file_path, "input_simus.txt"), comment="#", sep=r'\s+', header=None, usecols=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19], 
                        names=['event_idx', 'true_theta', 'true_phi', 'Primary_energy', 'Nature_primary', 'XmaxDistance', 'gramage', 'true_x_Xmax', 'true_y_Xmax', 'true_z_Xmax', 'true_x_core', 'true_y_core', 'true_z_core', 'core_alt', 'Number_triggered_antennas', 'inc', 'dec', 'mod'])
    
    if nmax is not None:
        df_temp = df_temp.iloc[:nmax]

    print(f"DataFrame built with {len(df_temp)} events from {file_path}/input_simus.txt")

    return df_temp

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
    event_idx_coord = np.loadtxt(position_file, usecols=0, dtype=int) # event index for each antenna entry
    event_idx_coord = np.unique(event_idx_coord) # unique event indices for antenna entries
    coords       = np.loadtxt(position_file, usecols=(1,2,3)) # coordinates of each antenna
    du_ids_coord = np.loadtxt(position_file, usecols=4, dtype=int) # antenna IDs (DU IDs) for each entry

    # --- Coïncidences ---
    event_idx_coinc, t_s, amp, du_ids_coinc = np.loadtxt(
        coinc_file, usecols=(0, 1, 2, 3), unpack=True
    ) # peak times in seconds and amplitudes for each coincidence entry

    event_idx_coinc = event_idx_coinc.astype(int) # event indices for each coincidence entry
    du_ids_coinc = du_ids_coinc.astype(int) # antenna IDs (DU IDs) for each coincidence entry
    t_m = t_s * pr.c_light

    events_ids = np.unique(event_idx_coinc)
    print(len(events_ids), "unique events in coincidence data.")
    
    good_events = [u for u in events_ids if np.sum(event_idx_coinc==u) >= 2] # events with at least 2 antennas in coincidence
    nco = len(good_events)

    nants = np.array([np.sum(event_idx_coinc==u) for u in good_events], dtype=int)  # number of antennas in coincidence for each good event
    nmax_ants = int(nants.max()) # maximum number of antennas in coincidence across all good events

    co_ant_idx = np.zeros((nco, nmax_ants),    dtype=int) # antenna indices for each coincidence, initialized to zero
    co_ant_coo = np.zeros((nco, nmax_ants, 3), dtype=np.float64) # antenna coordinates for each coincidence, initialized to zero
    co_evt_idx = np.zeros((nco, nmax_ants),    dtype=int) # event indices for each antenna in each coincidence, initialized to zero
    co_pti_met = np.zeros((nco, nmax_ants),    dtype=np.float64) # peak times in meters for each antenna in each coincidence, initialized to zero
    co_pti_sec = np.zeros((nco, nmax_ants),    dtype=np.float64) # peak times in seconds for each antenna in each coincidence, initialized to zero
    co_pea_amp = np.zeros((nco, nmax_ants),    dtype=np.float64) # peak amplitudes for each antenna in each coincidence, initialized to zero

    du_to_coord_idx = {}
    for i, du in enumerate(du_ids_coord):
        if du not in du_to_coord_idx:
            du_to_coord_idx[du] = i

    for k, u in enumerate(good_events):
        mask = (event_idx_coinc == u)
        n_ants = int(nants[k])

        du_event = du_ids_coinc[mask]

        # indices de ligne dans coords pour ces DU_id
        coord_indices = np.array([du_to_coord_idx[du] for du in du_event], dtype=int)

        co_ant_idx[k, :n_ants] = coord_indices
        co_ant_coo[k, :n_ants] = coords[coord_indices]
        co_evt_idx[k, :n_ants]  = u
        co_pti_met[k, :n_ants] = t_m[mask] - t_m[mask].min()
        co_pti_sec[k, :n_ants] = t_s[mask]
        co_pea_amp[k, :n_ants] = amp[mask]

    # --- Sauvegardes ---
    np.save(data_filepath + "/an_event_indices.npy", event_idx_coord)
    np.save(data_filepath + "/an_coordinates.npy", coords)
    np.save(data_filepath + "/an_du_ids.npy", du_ids_coord)
    np.save(data_filepath + "/an_nants.npy", len(coords))

    np.save(data_filepath + "/co_ncoincs.npy", np.array([nco], dtype=float))
    np.save(data_filepath + "/co_nants.npy", nants)
    np.save(data_filepath + "/co_nantsmax.npy", nmax_ants)
    np.save(data_filepath + "/co_antenna_index_array.npy", co_ant_idx)
    np.save(data_filepath + "/co_antenna_coords_array.npy", co_ant_coo)
    np.save(data_filepath + "/co_coinc_index_array.npy", co_evt_idx)
    np.save(data_filepath + "/co_peak_time_array.npy", co_pti_met)
    np.save(data_filepath + "/co_peak_time_array_in_s.npy", co_pti_sec)
    np.save(data_filepath + "/co_peak_amp_array.npy", co_pea_amp)

def old_npy_files_builder(file_path: str, data_filepath: str) -> None:
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
    np.save(data_filepath+"/an_event_indices.npy",idx)
    np.save(data_filepath+"/an_coordinates.npy",coords)
    np.save(data_filepath+"/an_du_ids.npy",init)
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