import wavefronts.params_config as pr
import pandas as pd
import jax.numpy as jnp


def build_K_vector(theta, phi):
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

    K_source = -1 * build_K_vector(alpha, beta)

    X_source = jnp.array([
        -rxmax * K_source[0],
        -rxmax * K_source[1],
        -rxmax * K_source[2] + pr.groundAltitude
    ])

    return jnp.array([X_source[0], X_source[1], X_source[2]])