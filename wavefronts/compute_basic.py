import wavefronts.params_config as pr
import jax.numpy as jnp
import numpy as np


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

    K_source = -1 * build_K_vector_jnp(alpha, beta)

    X_source = jnp.array([
        -rxmax * K_source[0],
        -rxmax * K_source[1],
        -rxmax * K_source[2] + pr.groundAltitude
    ])

    return jnp.array([X_source[0], X_source[1], X_source[2]])

def build_Xsource_np(alpha_rad: float, beta_rad: float, r_xmax: float) -> np.ndarray:
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

def build_K_vector_np(theta_rad: float, phi_rad: float) -> np.ndarray:
    """ Build the shower direction vector K from spherical coordinates
    Inputs:
        theta_rad: zenith angle in radians
        phi_rad: azimuthal angle in radians
    Outputs:
        K: shower direction vector
    """
    st = np.sin(theta_rad)
    ct = np.cos(theta_rad)
    sp = np.sin(phi_rad)
    cp = np.cos(phi_rad)

    K = np.array([
        -st * cp,
        -st * sp,
        -ct
    ], dtype=np.float64)

    return K / np.linalg.norm(K)  # Ensure K is a unit vector

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

