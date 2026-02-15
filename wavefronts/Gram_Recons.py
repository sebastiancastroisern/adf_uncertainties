import jax
import jax.numpy as jnp
import tqdm
import wavefronts.params_config as pr
import numpy as np
import pymap3d as pm

def compute_Xmax(SWF_rad: jnp.ndarray, ground_altitude: float = pr.groundAltitude) -> jnp.ndarray:
    """
    SWF_rad: (N, m), on utilise:
      - SWF_rad[:, 0] : angle alpha
      - SWF_rad[:, 1] : angle beta
      - SWF_rad[:, 3] : distance
    ground_altitude: altitude du sol en cm
    Retourne: Xmax de shape (N, 3)
    """
    SWF_rad = jnp.asarray(SWF_rad)

    alpha = SWF_rad[:, 0]
    beta  = SWF_rad[:, 1]
    dist  = SWF_rad[:, 3]

    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    cb, sb = jnp.cos(beta),  jnp.sin(beta)

    Xmax_vect = jnp.stack([sa * cb, sa * sb, ca], axis=1)
    origin = jnp.array([[0.0, 0.0, ground_altitude]])
    Xmax = Xmax_vect * dist[:, None] + origin

    return Xmax

compute_Xmax_jit = jax.jit(compute_Xmax)


def conversion_to_enu_numpy(Xmax: np.ndarray, lat0: float=pr.lat_0, lon0: float=pr.long_0) -> np.ndarray:
    """
    Conversion of Xmax from cartesian coordinates to altitude (ENU) using pymap3d, for a given reference latitude and longitude.
    Inputs:
        Xmax: np.ndarray
            Array of Xmax positions in cartesian coordinates (shape: (N, 3))
        lat0: float
            Reference latitude for ENU conversion (default: pr.lat_0)
        lon0: float
            Reference longitude for ENU conversion (default: pr.long_0)
    Outputs:
        H: np.ndarray
            Array of altitudes corresponding to Xmax positions (shape: (N,)) in cm
    """
    Xmax = np.asarray(Xmax)
    e = -Xmax[:, 1]
    n =  Xmax[:, 0]
    u =  Xmax[:, 2]

    H_list = []
    for ei, ni, ui in zip(e, n, u):
        _, _, H = pm.enu2geodetic(ei, ni, ui, lat0=lat0, lon0=lon0, h0=0.0)
        H_list.append(H)

    H = np.asarray(H_list) * 1e2  # m -> cm
    return H

def compute_grammage_numpy(Xmax_heights:np.ndarray, SWF_deg:np.ndarray, std_atm) -> np.ndarray:
    """ Compute grammage from Xmax heights and SWF parameters in degrees using a standard atmosphere model.
    Inputs:
        Xmax_heights: np.ndarray
            Array of Xmax heights in cm
        SWF_deg: np.ndarray
            Array of SWF parameters [theta, phi, R_source] in degrees and meters
        std_atm: object
            Standard atmosphere model with method h2X(height) to compute grammage from height
    Outputs:        
        grammages: np.ndarray
            Array of grammages corresponding to Xmax heights (shape: (N,)) in g/cm^2 
    """
    
    SWF_deg = np.asarray(SWF_deg)

    grammages = []
    for h, theta in tqdm.tqdm(zip(Xmax_heights, SWF_deg[:, 0]), total=len(Xmax_heights), desc="Computing grammages"):
        std_atm.set_theta(float(theta))
        grammages.append(std_atm.h2X(float(h)))

    return np.asarray(grammages)

