import jax.numpy as jnp
import jax
import MCEq.geometry.density_profiles as dp
import wavefronts.params_config as pr
import pandas as pd

# get the correction coefficients from the CSV file
df_coeffs = pd.read_csv('wavefronts/correction_coefficients.csv')
df_coeffs = df_coeffs[df_coeffs['value'] != 0.0]
jpn_coeffs = jnp.array(df_coeffs['value'].values)

# get the standard atmosphere density profile in jax
std_atm = dp.CorsikaAtmosphere('USStd')
altitudes_cm = jnp.logspace(5, 7, 10000)
densities_table = jnp.array([std_atm.get_density(alt) for alt in altitudes_cm]) # in g/cm^3
    
def density_jax(alt_cm: float) -> float:
    """Interpolate density at given altitude in cm using precomputed table.
    
    Inputs:
        alt_cm: float
            Altitude of the Xsource point in cm
    Outputs:
        desnsity: float
            Density value at altitude atl_cm in g/cm^3"""
    
    return jnp.interp(alt_cm, altitudes_cm, densities_table)

def jax_poly_features_3(sinalpha: float, rho: float) -> float:
    """Generate polynomial features up to degree 3 for sinalpha and rho.
    Inputs:
        sinalpha: float
            Sine of the geomagnetic angle
        rho: float
            Density at Xsource in kg/m^3
    Outputs:
        features: jnp.array
            Array of polynomial features up to degree 3 """
    
    return jnp.array([
        1,
        sinalpha,
        rho,
        sinalpha**2,
        sinalpha * rho,
        rho**2,
        sinalpha**3,
        (sinalpha**2) * rho,
        sinalpha * (rho**2),
        rho**3
    ])

def jax_linreg_predict(features: jnp.array, coef: jnp.array) -> float:
    """Predict using linear regression coefficients.
    Inputs:
        features: jnp.array
            Array of polynomial features
        coef: jnp.array
            Array of linear regression coefficients
    Outputs:
        prediction: float
            Predicted value of A_norm"""
    
    return jnp.dot(coef, features)

def build_K_vector(theta, phi):
    """ Build K vector from zenith and azimuth angles in radians 
    ️Inputs:
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

def jax_sin_alpha(theta, phi, B_norm=pr.B_vec_norm):
    """ Calculate sin alpha between shower axis and geomagnetic field, parameters in radians
    Inputs:
        theta: float
            Zenith angle in radians
        phi: float
            Azimuth angle in radians
        B_norm: jnp.array
            Normalized geomagnetic field vector
    Outputs:
        sin_alpha: float
            Sine of the angle between shower axis and geomagnetic field """

    K = build_K_vector(theta, phi)

    return jnp.linalg.norm(jnp.cross(K, B_norm))

def jax_build_Xsource(SWF_rad):
    """ Build source position (cartesian) from SWF parameters in radians
    Inputs:
        SWF_rad: jnp.array
            Array of SWF parameters [theta, phi, R_source] in radians and meters
    Outputs:
        X_source: jnp.array
            Source position in cartesian coordinates [X, Y, Z] in meters """

    alpha, beta = SWF_rad[0], SWF_rad[1]
    K_source = build_K_vector(alpha, beta)
    return jnp.array([
        -SWF_rad[2] * K_source[0],
        -SWF_rad[2] * K_source[1],
        -SWF_rad[2] * K_source[2] + pr.groundAltitude
    ])

def jax_altitude(SWF_rad: jnp.array):
    """Calculate altitude of the source from SWF parameters in radians
    Inputs:
        SWF_rad: jnp.array
            Array of SWF parameters [theta, phi, R_source, t_s] in radians, meters and seconds
    Outputs:
        altitude: float
            Altitude of the source in meters """

    X_source = jax_build_Xsource(SWF_rad)
    R2 = X_source[0]**2 + X_source[1]**2

    return jnp.sqrt(R2 + (X_source[2] + pr.R_earth)**2) - pr.R_earth

def jax_energy_recons(recons_rad: jnp.array, jpn_coeffs: jnp.array) -> float:
    """ Reconstruct energy from radio wavefront parameters in radians
    Inputs:
        recons_rad: jnp.array
            Array of reconstructed parameters [theta, phi, R_source, t_s, ADF parameters] in radians, meters and seconds
        jpn_coeffs: jnp.array
            Array of linear regression coefficients
    Outputs:
        energy: float
            Reconstructed energy in eV"""

    SWF = recons_rad[:4]
    ADF = recons_rad[4:]

    Xsource_alt = jax_altitude(SWF)
    rho_Xsource_kg_m3 = density_jax(Xsource_alt * 1e2) * 1e3  # convert g/cm^3 to kg/m^3

    sinalpha = jax_sin_alpha(ADF[0], ADF[1])

    features = jax_poly_features_3(sinalpha, rho_Xsource_kg_m3)

    A_norm_pred = jax_linreg_predict(features, jpn_coeffs)

    energy = ADF[3] / (A_norm_pred * sinalpha)

    return energy

def jax_energy_and_uncertainty(recons_rad: jnp.array, cov_mat: jnp.array, jnp_coeffs: jnp.array) -> jnp.array:
    """
    Computes the energy reconstruction and its uncertainty using the JAX framework.
    Inputs:
        recons_rad: jnp.array
            Array of reconstructed parameters [theta, phi, R_source, t_s, ADF parameters] in radians, meters and seconds
        cov_mat: jnp.array
            Covariance matrix of the reconstructed parameters
        jnp_coeffs: jnp.array
            Array of linear regression coefficients
    Outputs:
        energy_and_uncertainty: jnp.array
            Array containing the reconstructed energy and its uncertainty [energy, uncertainty]
    """
    
    energy = jax_energy_recons(recons_rad, jnp_coeffs)

    def energy_func(recons_rad_flat):
        recons_rad_reshaped = recons_rad_flat.reshape(recons_rad.shape)
        return jax_energy_recons(recons_rad_reshaped, jnp_coeffs)

    # gradient
    energy_grad = jax.grad(energy_func)(recons_rad)
    
    # hessian
    energy_hess = jax.hessian(energy_func)(recons_rad)
    
    # first order term
    variance_1st = jnp.dot(energy_grad, jnp.dot(cov_mat, energy_grad))
    
    # second order term
    variance_2nd = 0.5 * jnp.trace(jnp.dot(jnp.dot(energy_hess, cov_mat), 
                                            jnp.dot(energy_hess, cov_mat)))
    
    variance = variance_1st + variance_2nd
    energy_uncertainty = jnp.sqrt(jnp.abs(variance)) 

    return jnp.array([energy, energy_uncertainty])