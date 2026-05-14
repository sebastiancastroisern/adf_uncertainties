import wavefronts.params_config as pr
import jax.numpy as jnp
import numpy as np


# ======================
# Physical constants
# ======================

c_light = 2.997924580e8 # m/s
R_earth = 6371007.0     # m
lat_0 = 0.99  
long_0 = 93.94   
groundAltitude = 1264.0 # m


# ======================
# Atmospheric model
# ======================

ns = 325
kr = -0.1218
rho0 = 1.225  # kg/m^3
h0 = 8_000.0  # m
cr = 1.0 

# ======================
# Magnetic field
# ======================

# Values at the Xiaodushan Observatory, China (for the DC2 simulations)
modulus = 56.482  # uT (microtesla)
B_inc = np.deg2rad(61.6)  # degrés
B_dec = np.deg2rad(0.1253)  # degrés

B_vec_norm = np.array([
    np.sin(B_inc) * np.cos(B_dec),
    np.sin(B_inc) * np.sin(B_dec),
    np.cos(B_inc),
]) / np.linalg.norm(np.array([
    np.sin(B_inc) * np.cos(B_dec),
    np.sin(B_inc) * np.sin(B_dec),
    np.cos(B_inc),
]))

B_vec = B_vec_norm * modulus * 1e-6  # en T (tesla)

# ======================
# Interpolation
# ======================

n_omega_cr = 20
pickle_file = "wavefronts/correction_coefficients.pkl"
csv_coeff_corr = "wavefronts/correction_coefficients.csv"
npy_cov_matrix = "wavefronts/cov_beta.npy"

# ======================
# Bounds (single source of truth)
# Order: alpha, beta, rxmax, t0, theta, phi, dw, A
# ======================

bound_alpha = [0.0, np.pi]
bound_beta  = [0.0, 2.0 * np.pi]
bound_rxmax = [0.0, 1e6]
bound_t0    = [-1e-2, 0.0]
bound_theta = [0.0, np.pi]
bound_phi   = [0.0, 2.0 * np.pi]
bound_dw    = [1e-2, 50.0]
bound_A     = [1e3, 1e12]

bounds = np.array([
    bound_alpha,
    bound_beta,
    bound_rxmax,
    bound_t0,
    bound_theta,
    bound_phi,
    bound_dw,
    bound_A,
    ])

# ======================
# Sigmas for proposal distribution in MCMC
# ======================

sigmas = np.array([
    1*np.pi/180,  # alpha
    1*np.pi/180,  # beta
    1e3,          # rxmax
    1e3,          # t0
    1*np.pi/180,  # theta
    1*np.pi/180,  # phi
    1.0,          # dw
    1e6,          # A
    ])  

def build_K_vector_jnp(theta, phi):
    """ Build K vector from zenith and azimuth angles in radians 
    Inputs:
        theta: float
            Zenith angle in radians
        phi: float
            Azimuth angle in radians
    Outputs:
        K: jnp.array
            K vector as a jnp.array [Kx, Ky, Kz] """

    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    sp = jnp.sin(phi)
    cp = jnp.cos(phi)

    return -1 * jnp.array([st * cp, st * sp, ct])

def compute_Xsource(alpha: float, beta: float, rxmax: float) -> tuple:
    """Compute the Xsource position in 3D space based on the reconstructed parameters.
    
    Inputs:
        alpha: float
            Reconstructed alpha parameter (in rad)
        beta: float
            Reconstructed beta parameter (in rad)
        rxmax: float
            Reconstructed rxmax parameter (in meters)
        t0: float
            Reconstructed t0 parameter (in seconds)
    Outputs:        
        tuple:
            A tuple containing the Xsource position in cartesian coordinates (X, Y, Z) in meters
    """

    K_source = build_K_vector_jnp(alpha, beta)

    X_source = jnp.array([
        -rxmax * K_source[0],
        -rxmax * K_source[1],
        -rxmax * K_source[2] + groundAltitude
    ])

    return jnp.array([X_source[0], X_source[1], X_source[2]])

def compute_B_vec(inc, dec, mod=1.0):
    """
    Compute the magnetic field vector(s) B in the local coordinate system.
    Inputs:
        inc: float or np.ndarray
            Inclination angle(s) in degrees
        dec: float or np.ndarray
            Declination angle(s) in degrees
        mod: float or np.ndarray, optional
            Modulus of the magnetic field in microtesla (default is 1.0, which means no scaling)
    Outputs:        
        np.ndarray
            Magnetic field vector(s) in the local coordinate system, with shape (..., 3) where the last dimension corresponds to the (Bx, By, Bz) components in tesla
    """
    inc = np.asarray(inc, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    mod = np.asarray(mod, dtype=np.float64)

    mod_T = np.where(mod == 1.0, mod, mod * 1e-6)

    inc_rad = np.deg2rad(inc)
    dec_rad = np.deg2rad(dec)

    Bx = mod_T * np.cos(inc_rad) * np.cos(dec_rad)
    By = mod_T * np.cos(inc_rad) * np.sin(dec_rad)
    Bz = mod_T * np.sin(inc_rad)

    return np.stack([Bx, By, Bz], axis=-1)

def uncertainties_from_file_path(file_path: str) -> tuple:
    """Determine the uncertainties (time jitter, background noise, amplitude uncertainty) based on the file path.
    This is a heuristic approach to assign realistic uncertainties to the reconstructed parameters based on the type of data (e.g., Efield, GP300, GP289, CoREAS) and the type of ADC (NJ or AN) used in the simulation or measurement. The values are chosen based on typical characteristics of these data types and can be adjusted as needed.

    Inputs:
        file_path: str
            The file path of the data, which is used to determine the type of data and assign appropriate uncertainties.
    Outputs:
        tuple:
            A tuple containing the following uncertainties:
            - min_amplitude: float
                The minimum amplitude increment (e.g., 1 ADC count or 1e-3 µV/m) that can be resolved in the data.
            - jitter_time: float
                The time jitter (in seconds) to be added to the reconstructed parameters to account for timing uncertainties.
            - background_noise: float
                The standard deviation of the background noise (in ADC counts or µV/m) to be added to the reconstructed parameters to account for noise in the data.
            - amplitude_uncertainty: float
                The relative uncertainty on the amplitude (e.g., due to calibration) to be applied to the reconstructed parameters.
    """

    jitter_time_min = 0.5e-9 

    # Determine noise floor and amplitude uncertainty based on file path
    is_efield = 'efield' in file_path
    is_gp300 = 'GP300' in file_path
    is_gp289 = 'GP289' in file_path
    is_nj_adc = '-NJ_adc' in file_path
    is_an_adc = '-AN_adc' in file_path

    if is_efield:
        min_amplitude    = 1e-3 # 1e-3 µV/m, minimal increment of values
        jitter_time      = 0.0 # No added time jitter en Efield
        background_noise = 0.0 # Std of background noise
        amplitude_uncertainty = 0.0 # Relative uncertainty on amplitude, e.g. due to calibration (0.075 = 7.5%)

    elif is_gp300:  # ZHAireS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_nj_adc:
            jitter_time = 0.0
            background_noise = 0.0
            amplitude_uncertainty = 0.0
        elif is_an_adc:
            jitter_time = 10e-9
            background_noise = 15.0
            amplitude_uncertainty = 0.075
        else:
            jitter_time = 10e-9
            background_noise = 4.0
            amplitude_uncertainty = 0.075

    elif is_gp289:  # ZHAireS
        min_amplitude = 1.0 
        if is_nj_adc:
            jitter_time = 0.0
            background_noise = 0.0
            amplitude_uncertainty = 0.0
        elif is_an_adc:
            jitter_time = 10e-9
            background_noise = 12.0
            amplitude_uncertainty = 0.075
        else:
            jitter_time = 10e-9
            background_noise = 5.0
            amplitude_uncertainty = 0.075

    else:  # CoREAS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_an_adc:
            jitter_time = 10e-9
            background_noise = 10.0
            amplitude_uncertainty = 0.075
    
    sigma_time = np.sqrt(jitter_time**2 + jitter_time_min**2)
    background_noise = np.sqrt(background_noise**2 + min_amplitude**2)

    return (sigma_time, background_noise, amplitude_uncertainty)
