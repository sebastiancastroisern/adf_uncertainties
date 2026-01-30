# params.py
import numpy as np

# ======================
# Physical constants
# ======================

c_light = 2.997924580e8 # m/s
R_earth = 6371007.0     # m
groundAltitude = 1264.0 # m

# ======================
# Atmospheric model
# ======================

ns = 325
kr = -0.1218
rho0 = 1.225  # kg/m^3
h0 = 8_000.0  # m

# ======================
# Magnetic field
# ======================

# Values at the Xiaodushan Observatory, China

modulus = 56.482  # uT (microtesla)
B_inc = 61.6  # degrés
B_dec = 0.1253  # degrés

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

# # For the no-noise simulation
# B_dec = 0.
# B_inc = np.pi/2. + 1.0609856522873529
# # Magnetic field direction (unit) vector
# Bvec = np.array([np.sin(B_inc)*np.cos(B_dec),np.sin(B_inc)*np.sin(B_dec),np.cos(B_inc)])

# B_vec_norm = Bvec / np.linalg.norm(Bvec)
# B_vec = B_vec_norm

# ======================
# Noise / detector
# ======================

jitter_time = 5e-9                # s
galactic_noise_floor = 8.0        # µV (aucune idée j'ai pas vérifié)
assym_coeff = 0.01
cr = 1.0

# ======================
# Interpolation
# ======================

n_omega_cr = 20

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
bound_dw    = [0.1, 100.0]
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
