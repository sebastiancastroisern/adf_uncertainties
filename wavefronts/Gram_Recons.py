import jax
import jax.numpy as jnp
import tqdm
import wavefronts.params_config as pr
import numpy as np
import pandas as pd

# Load Linsley atmospheric density model (height vs density)
linsey_atmosphere = pd.read_parquet("wavefronts/linsley_atmosphere.parquet")

def compute_k_vect(alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """Compute the direction unit vector of incoming cosmic ray (k vector).
    
    Converts spherical angles (alpha, beta) to cartesian coordinates.
    
    Args:
        alpha: Zenith angle in radians
        beta: Azimuth angle in radians
    
    Returns:
        Direction vector in cartesian coordinates [kx, ky, kz]
    """
    
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    cb, sb = jnp.cos(beta),  jnp.sin(beta)

    k_vect = jnp.array([sa * cb, sa * sb, ca])
    return k_vect

def compute_Xsource(SWF_rad: jnp.ndarray, ground_altitude: float = pr.groundAltitude) -> jnp.ndarray:
    """Compute source position from spherical wavefront parameters.
    
    Combines direction angles and distance to locate the cosmic ray source
    in cartesian coordinates relative to ground level.
    
    Args:
        SWF_rad: Spherical wavefront parameters (Nx4 array)
            [:, 0] = alpha (zenith angle) in radians
            [:, 1] = beta (azimuth angle) in radians
            [:, 2] = distance from ground in m
        ground_altitude: Reference altitude (ground level) in m
    
    Returns:
        Source positions in cartesian coordinates (Nx3 array) in m
    """
    SWF_rad = jnp.asarray(SWF_rad)

    alpha = SWF_rad[0]
    beta  = SWF_rad[1]
    dist  = SWF_rad[2]

    Xsource_vect = compute_k_vect(alpha, beta)
    origin = jnp.array([[0.0, 0.0, ground_altitude]])
    Xsource = Xsource_vect * dist + origin
    Xsource = Xsource.reshape(3)  # Reshape to (3,) for consistency

    return Xsource

def jax_altitude(X_source: jnp.ndarray, R_earth: float= pr.R_earth) -> jnp.ndarray:
    """X_source : (3,) en mètres, retourne altitude en cm"""

    # Distance horizontale au carré
    R2 = X_source[0]**2 + X_source[1]**2

    return (jnp.sqrt(R2 + (X_source[2] + R_earth)**2) - R_earth) * 1e2

def jax_altitude_multi(X_point: jnp.ndarray, R_earth: float= pr.R_earth) -> jnp.ndarray:
    """X_point : (...,3) en mètres, retourne altitude en cm"""
    
    # Distance horizontale au carré
    R2 = X_point[:, 0]**2 + X_point[:, 1]**2

    return (jnp.sqrt(R2 + (X_point[:, 2] + R_earth)**2) - R_earth) * 1e2


def find_max_alt_point(Xsource_heights: jnp.array, theta_rad: float, max_altitude_cm: float = 110e5) -> jnp.array:
    """Find distance to atmosphere boundary along ray trajectory.
    
    Solves sphere-ray intersection to find where cosmic ray exits the atmosphere
    (at specified maximum altitude). Uses positive solution for upward rays.
    
    Args:
        Xsource_heights: Source altitude in cm
        theta_rad: Zenith angle in radians
        max_altitude_cm: Atmosphere upper boundary in cm (default: 110 km)
    
    Returns:
        Distance along ray direction to atmosphere boundary in cm
    """

    total_radius = pr.R_earth * 1e2 + max_altitude_cm
    distance = - (Xsource_heights + pr.R_earth * 1e2) * jnp.cos(theta_rad) + jnp.sqrt(total_radius**2 - ((Xsource_heights + pr.R_earth * 1e2) * jnp.sin(theta_rad))**2)

    return distance

def jax_slant_depth_adf(SWF_rad:jnp.ndarray, ADF_rad:jnp.ndarray) -> jnp.ndarray:
    """Compute atmospheric slant depth along cosmic ray trajectory.
    
    Integrates atmospheric density along the ray path from source to 
    atmosphere boundary using Linsley atmospheric model for one event.
    
    Args:
        SWF_rad: Spherical wavefront parameters [alpha, beta, ...] in radians
        ADF_rad: Additional parameters 
    
    Returns:
        Total slant depth in g/cm²
    """
    num_points = 10000  # Number of sampling points along the ray path

    # Get source position in cartesian coordinates
    Xsource = compute_Xsource(SWF_rad)
    # print("Source position (m):", Xsource)
    # Get source altitude
    height_cm = jax_altitude(Xsource)

    # Find distance to atmosphere boundary along ray direction
    max_alt_point_dist_cm = find_max_alt_point(height_cm, ADF_rad[0]) # send the true zenith angle
    max_alt_point_dist_m = max_alt_point_dist_cm * 1e-2  # convert cm to m

    # Generate sampling points along ray path
    K_vect = compute_k_vect(ADF_rad[0], ADF_rad[1]) # use the adf angles
    Max_point = Xsource + K_vect * max_alt_point_dist_m
    line_points = Max_point + jnp.linspace(0, 1, num_points)[:, None] * (Xsource - Max_point)

    # Evaluate altitude and density at each point
    heights_along_line_cm = jax_altitude_multi(line_points)
    densities = jnp.interp(heights_along_line_cm, linsey_atmosphere['height_cm'].values, linsey_atmosphere['density_g_cm3'].values, right=0.0)
    
    # Integrate density along path
    delta_dist = max_alt_point_dist_cm / (num_points - 1)
    slant_depth = jnp.sum(densities, axis=0) * delta_dist # g/cm^2

    return slant_depth

def jax_slant_depth(SWF_rad:jnp.ndarray) -> jnp.ndarray:
    """
    OLD VERSION NOT USED IN CRB_NEX.PY
    Compute atmospheric slant depth along cosmic ray trajectory using the SWF alpha angle as a proxy for the shower angle.
    
    Integrates atmospheric density along the ray path from source to 
    atmosphere boundary using Linsley atmospheric model for one event.
    
    Args:
        SWF_rad: Spherical wavefront parameters [alpha, beta, ...] in radians
    
    Returns:
        Total slant depth in g/cm²
    """
    num_points = 10000  # Number of sampling points along the ray path

    # Get source position in cartesian coordinates
    Xsource = compute_Xsource(SWF_rad)
    # print("Source position (m):", Xsource)
    # Get source altitude
    height_cm = jax_altitude(Xsource)

    # Find distance to atmosphere boundary along ray direction
    max_alt_point_dist_cm = find_max_alt_point(height_cm, SWF_rad[0])
    max_alt_point_dist_m = max_alt_point_dist_cm * 1e-2  # convert cm to m

    # Generate sampling points along ray path
    K_vect = compute_k_vect(SWF_rad[0], SWF_rad[1])
    Max_point = Xsource + K_vect * max_alt_point_dist_m
    line_points = Max_point + jnp.linspace(0, 1, num_points)[:, None] * (Xsource - Max_point)

    # Evaluate altitude and density at each point
    heights_along_line_cm = jax_altitude_multi(line_points)
    densities = jnp.interp(heights_along_line_cm, linsey_atmosphere['height_cm'].values, linsey_atmosphere['density_g_cm3'].values, right=0.0)
    
    # Integrate density along path
    delta_dist = max_alt_point_dist_cm / (num_points - 1)
    slant_depth = jnp.sum(densities, axis=0) * delta_dist # g/cm^2

    return slant_depth

jax_slant_depth_jit = jax.jit(jax_slant_depth)
jax_slant_depth_adf_jit = jax.jit(jax_slant_depth_adf)

def depth_true_xmax(Xsource:jnp.ndarray, theta_phi_rad:jnp.ndarray) -> jnp.ndarray:
    """Compute atmospheric slant depth along cosmic ray trajectory.
    
    Integrates atmospheric density along the ray path from source to 
    atmosphere boundary using Linsley atmospheric model for one event.
    
    Args:
        Xsource: Source position in cartesian coordinates
        theta_phi_rad: Zenith and azimuth angles in radians
    
    Returns:
        Total slant depth in g/cm²
    """
    num_points = 10000  # Number of sampling points along the ray path

    # print("Source position (m):", Xsource)
    # Get source altitude
    height_cm = jax_altitude(Xsource)

    # Find distance to atmosphere boundary along ray direction
    max_alt_point_dist_cm = find_max_alt_point(height_cm, theta_phi_rad[0]) # send the true zenith angle
    max_alt_point_dist_m = max_alt_point_dist_cm * 1e-2  # convert cm to m

    # Generate sampling points along ray path
    K_vect = compute_k_vect(theta_phi_rad[0], theta_phi_rad[1]) # use the theta_phi angles
    Max_point = Xsource + K_vect * max_alt_point_dist_m
    line_points = Max_point + jnp.linspace(0, 1, num_points)[:, None] * (Xsource - Max_point)

    # Evaluate altitude and density at each point
    heights_along_line_cm = jax_altitude_multi(line_points)
    densities = jnp.interp(heights_along_line_cm, linsey_atmosphere['height_cm'].values, linsey_atmosphere['density_g_cm3'].values, right=0.0)
    
    # Integrate density along path
    delta_dist = max_alt_point_dist_cm / (num_points - 1)
    slant_depth = jnp.sum(densities, axis=0) * delta_dist # g/cm^2

    return slant_depth

depth_true_xmax_jit = jax.jit(depth_true_xmax)