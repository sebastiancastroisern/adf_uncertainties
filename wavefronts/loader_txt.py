import os
import jax.numpy as jnp
import numpy as np
import pandas as pd
import wavefronts.params_config as pr
import wavefronts.energy_jax as nrjax
import wavefronts.compute_basic as cmpt

def load_data_files_jnp(file_path: str, n_to_process: int=None) -> tuple:
    """ Load data files from numpy files using JAX.
    Inputs:
        file_path: str
            Path to the directory containing the result files
        n_to_process: int, optional
            Number of coincidences to process (default is None, meaning all)
    Outputs:
        tuple:
            A tuple containing nants, antenna_coords_array, peak_time_array_m, peak_time_array_s, peak_amp_array, ncoincs
    """
    data_file_path = os.path.join(file_path,'data_npy')
    nants                = jnp.load(os.path.join(data_file_path,'co_nants.npy'))
    antenna_coords_array = jnp.load(os.path.join(data_file_path,'co_antenna_coords_array.npy'))
    peak_time_array_m    = jnp.load(os.path.join(data_file_path,'co_peak_time_array.npy'))
    peak_time_array_s    = jnp.load(os.path.join(data_file_path,'co_peak_time_array_in_s.npy'))
    peak_amp_array       = jnp.load(os.path.join(data_file_path,'co_peak_amp_array.npy'))
    ncoincs              = int(jnp.load(os.path.join(data_file_path,'co_ncoincs.npy'))[0])

    return (nants, antenna_coords_array, peak_time_array_m, peak_time_array_s, peak_amp_array, ncoincs)

def load_data_files_np(file_path: str, n_to_process: int=None) -> tuple:
    """ Load data files from numpy files.
    Inputs:
        file_path: str
            Path to the directory containing the result files
            n_to_process: int, optional
    Outputs:
        tuple:
            A tuple containing nants, antenna_coords_array, peak_time_array_m, peak_time_array_s, peak_amp_array, ncoincs
    """
    data_file_path = os.path.join(file_path,'data_npy')
    nants                = np.load(os.path.join(data_file_path,'co_nants.npy'))
    antenna_coords_array = np.load(os.path.join(data_file_path,'co_antenna_coords_array.npy'))
    peak_time_array_m    = np.load(os.path.join(data_file_path,'co_peak_time_array.npy'))
    peak_time_array_s    = np.load(os.path.join(data_file_path,'co_peak_time_array_in_s.npy'))
    peak_amp_array       = np.load(os.path.join(data_file_path,'co_peak_amp_array.npy'))
    ncoincs              = int(np.load(os.path.join(data_file_path,'co_ncoincs.npy'))[0])

    return (nants, antenna_coords_array, peak_time_array_m, peak_time_array_s, peak_amp_array, ncoincs)

def load_adf_res_jnp(file_path: str, n_to_process: int=None) -> tuple:
    """ Load ADF results from numpy files.
    Inputs:
        file_path: str
            Path to the directory containing the result files
    Outputs:
        tuple:
            A tuple containing SWF_res, ADF_res, and CRB_res dictionaries
    """

    SWF_res = jnp.load(os.path.join(file_path, 'SWF_res.npy'), allow_pickle=True).item()
    ADF_res = jnp.load(os.path.join(file_path, 'ADF_res.npy'), allow_pickle=True).item()
    CRB_res = jnp.load(os.path.join(file_path, 'CRB_res.npy'), allow_pickle=True).item()
    CRB_cov = jnp.load(os.path.join(file_path, 'CRB_fisher_matrices.npy'), allow_pickle=True).item()
    SWF_res = SWF_res['data']
    ADF_res = ADF_res['data']
    CRB_res = CRB_res['data']
    CRB_cov = CRB_cov['data']
    
    if n_to_process is None:
        n_to_process = len(SWF_res)

    return (SWF_res[:n_to_process], ADF_res[:n_to_process], CRB_res[:n_to_process], CRB_cov[:n_to_process])

def load_adf_res_np(file_path: str, n_to_process: int=None) -> tuple:
    """ Load ADF results from numpy files.
    Inputs:
        file_path: str
            Path to the directory containing the result files
    Outputs:
        tuple:
            A tuple containing SWF_res, ADF_res, and CRB_res dictionaries
    """

    SWF_res = np.load(os.path.join(file_path, 'SWF_res.npy'), allow_pickle=True).item()
    ADF_res = np.load(os.path.join(file_path, 'ADF_res.npy'), allow_pickle=True).item()
    CRB_res = np.load(os.path.join(file_path, 'CRB_res.npy'), allow_pickle=True).item()
    CRB_cov = np.load(os.path.join(file_path, 'CRB_fisher_matrices.npy'), allow_pickle=True).item()
    SWF_res = SWF_res['data']
    ADF_res = ADF_res['data']
    CRB_res = CRB_res['data']
    CRB_cov = CRB_cov['data']
    
    if n_to_process is None:
        n_to_process = len(SWF_res)

    return (SWF_res[:n_to_process], ADF_res[:n_to_process], CRB_res[:n_to_process], CRB_cov[:n_to_process])  

def load_losses_np(file_path: str, n_to_process: int=None) -> tuple:
    """ Load ADF and SWF losses from numpy files.
    Inputs:
        file_path: str
            Path to the directory containing the result files
    Outputs:
        tuple:
            A tuple containing ADF_losses and SWF_losses
    """
    
    ADF_losses = np.load(os.path.join(file_path, 'ADF_res.npy'), allow_pickle=True).item()['loss']
    SWF_losses = np.load(os.path.join(file_path, 'SWF_res.npy'), allow_pickle=True).item()['loss']
    
    if n_to_process is None:
        n_to_process = len(ADF_losses)

    return (ADF_losses[:n_to_process], SWF_losses[:n_to_process])

def load_df(file_path: str, n_to_process: int=None) -> pd.DataFrame:
    """
    Load simulation input data from text file into a pandas DataFrame.
    Inputs:
        file_path: str
            Path to the directory containing the input_simus.txt file
        n_to_process: int
            Number of rows to read from the file
    Outputs:
        df_temp: pd.DataFrame
            DataFrame containing the simulation input data
    """

    if file_path.endswith('AN3-08_dec_25') or file_path.endswith('test_AN3') or file_path.endswith('test_NJ'):
        df_temp = pd.read_csv(os.path.join(file_path, "input_simus.txt"), comment="#", sep=r'\s+', header=None, usecols=[1, 2, 3, 5, 6, 7, 8, 9, 10, 15], 
                        names=['true_theta', 'true_phi', 'Primary_energy', 'Nature_primary', 'XmaxDistance', 'gramage', 'x_Xmax', 'y_Xmax', 'z_Xmax', 'Number_triggered_antennas'])
    else:
        df_temp = pd.read_csv(os.path.join(file_path, "input_simus.txt"), comment="#", sep=r'\s+', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
                        names=['true_theta', 'true_phi', 'Primary_energy', 'Em_energy', 'Nature_primary', 'XmaxDistance', 'gramage', 'x_Xmax', 'y_Xmax', 'z_Xmax', 'Number_triggered_antennas'])
        
    if n_to_process is not None:
        df_temp = df_temp.iloc[:n_to_process].copy()

    return df_temp

def build_full_df(file_path: str, n_to_process: int=None) -> pd.DataFrame:
    """
    Build the full DataFrame by loading simulation input data.
    Inputs:
        file_path: str
            Path to the directory containing the input_simus.txt file
        n_to_process: int
            Number of rows to read from the file
    Outputs:
        df_full: pd.DataFrame
            DataFrame containing the full simulation input data, adf, swf and crb results, including losses
    """

    # Load the simulation input data
    df_full = load_df(file_path, n_to_process)
    SWF_res, ADF_res, CRB_res, _ = load_adf_res_np(file_path, n_to_process)
    ADF_losses, SWF_losses = load_losses_np(file_path, n_to_process)
    
    df_full['ADF_loss_ndof'] = ADF_losses
    df_full['SWF_loss_ndof'] = SWF_losses
    
    # Add reconstructed parameters to the DataFrame
    new_cols = np.array([[ADF_res[i][0], ADF_res[i][1], ADF_res[i][2], ADF_res[i][3]] for i in range(len(ADF_res))])
    df_full[['recons_theta', 'recons_phi', 'recons_delta_omega', 'recons_amplitude']] = new_cols
    new_cols = np.array([[SWF_res[i][0], SWF_res[i][1], SWF_res[i][2], SWF_res[i][3]] for i in range(len(SWF_res))])
    df_full[['recons_alpha', 'recons_beta', 'recons_rxmax', 'recons_t0']] = new_cols
    new_cols = np.array([[CRB_res[i][0], CRB_res[i][1], CRB_res[i][2], CRB_res[i][3], CRB_res[i][4], CRB_res[i][5], CRB_res[i][6], CRB_res[i][7]] for i in range(len(CRB_res))])
    df_full[['stds_alpha', 'stds_beta', 'stds_rxmax', 'stds_t0', 'stds_theta', 'stds_phi', 'stds_delta_omega', 'stds_amplitude']] = new_cols

    return df_full


def build_df_energy(file_path: str, n_to_process: int=None) -> pd.DataFrame:
    """
    Build the full DataFrame by loading simulation input data, sin_alpha and air density at Xsource.
    Inputs:
        file_path: str
            Path to the directory containing the input_simus.txt file
        n_to_process: int
            Number of rows to read from the file
    Outputs:
        df_full: pd.DataFrame
            DataFrame containing the full simulation input data
    """

    # Load the simulation input data
    df_full = load_df(file_path, n_to_process)
    SWF_res, ADF_res, CRB_res, _ = load_adf_res_np(file_path, n_to_process)

    # Add reconstructed parameters to the DataFrame
    new_cols = np.array([[ADF_res[i][0], ADF_res[i][1], ADF_res[i][2], ADF_res[i][3]] for i in range(len(ADF_res))])
    df_full[['recons_theta', 'recons_phi', 'recons_delta_omega', 'recons_amplitude']] = new_cols
    new_cols = np.array([[SWF_res[i][0], SWF_res[i][1], SWF_res[i][2], SWF_res[i][3]] for i in range(len(SWF_res))])
    df_full[['recons_alpha', 'recons_beta', 'recons_rxmax', 'recons_t0']] = new_cols
    new_cols = np.array([[CRB_res[i][0], CRB_res[i][1], CRB_res[i][2], CRB_res[i][3], CRB_res[i][4], CRB_res[i][5], CRB_res[i][6], CRB_res[i][7]] for i in range(len(CRB_res))])
    df_full[['stds_alpha', 'stds_beta', 'stds_rxmax', 'stds_t0', 'stds_theta', 'stds_phi', 'stds_delta_omega', 'stds_amplitude']] = new_cols

    thetas_rad = np.deg2rad(df_full['recons_theta'].to_numpy())
    phis_rad = np.deg2rad(df_full['recons_phi'].to_numpy())
    df_full['sin_alpha'] = np.array([nrjax.jax_sin_alpha(theta, phi) for theta, phi in zip(thetas_rad, phis_rad)])


    alphas_rad = np.deg2rad(df_full['recons_alpha'].to_numpy())
    betas_rad = np.deg2rad(df_full['recons_beta'].to_numpy())
    rxmaxs = df_full['recons_rxmax'].to_numpy()
    t0s = df_full['recons_t0'].to_numpy()
    test = ([
        nrjax.density_jax(
            nrjax.jax_altitude(
                
                np.array([alphas_rad[i], betas_rad[i], rxmaxs[i], t0s[i]])
                
                )) * 1e3 for i in range(len(CRB_res))
    ])
    df_full[['rho_kg_m3']] = np.reshape(test, (-1,1))

    return df_full