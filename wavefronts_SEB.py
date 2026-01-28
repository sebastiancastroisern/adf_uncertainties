import numpy as np
from numba import njit, float64, prange
from scipy.spatial.transform import Rotation as R
from solver import newton
from rotation import rotation
from scipy.optimize import brentq
import params_config as pr

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}

# Simple numba example
@njit(**kwd)
def dotme(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    res =  np.dot(x,x)
    res += np.dot(y,y)
    res += np.dot(z,z)
    return(res)

@njit(**kwd)
def RefractionIndexAtPosition(X: np.ndarray) -> float:
    """Calculate the refraction index at a given position X using the configuration parameters."""
    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt( (X[2]+pr.R_earth)**2 + R2 ) - pr.R_earth)/1e3 # Altitude in km
    rh = pr.ns*np.exp(pr.kr*h)
    return 1.+1e-6*rh

@njit(**kwd)
def ZHSEffectiveRefractionIndex(X0: np.ndarray, Xa: np.ndarray) -> float:
    """Calculate the effective refraction index between emission point X0 and antenna position Xa using the configuration parameters."""
    # X0 : emission point
    # Xa : antenna position

    R02 = X0[0]**2 + X0[1]**2 # Radial distance squared at emission point
    
    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+pr.R_earth)**2 + R02 ) - pr.R_earth)/1e3
    
    # Refractivity at emission 
    # rh0 = ns*np.exp(kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        # nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        # nint = max(int(modr/5e4) + 1, 1)

        K = U/nint

        # Current point coordinates and altitude
        Curr  = X0
        currh = h0
        s = 0.

        for i in np.arange(nint):
            Next = Curr + K # Next point
            nextR2 = Next[0]*Next[0] + Next[1]*Next[1]
            nexth  = (np.sqrt( (Next[2]+pr.R_earth)**2 + nextR2 ) - pr.R_earth)/1e3
            if (np.abs(nexth-currh) > 1e-10):
                s += (np.exp(pr.kr*nexth)-np.exp(pr.kr*currh))/(pr.kr*(nexth-currh))
            else:
                s += np.exp(pr.kr*currh)
            Curr = Next
            currh = nexth

        avn = pr.ns*s/nint
        n_eff = 1. + 1e-6*avn # Effective (average) index

    else:

        # without numerical integration
        hd = Xa[2]/1e3 # Antenna altitude
        #if (np.abs(hd-h0) > 1e-10):
        avn = (pr.ns/(pr.kr*(hd-h0)))*(np.exp(pr.kr*hd)-np.exp(pr.kr*h0))
        #else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)

@njit(**kwd)
def compute_observer_position(omega: float, Xmax: np.ndarray, U: np.ndarray, K: np.ndarray, xmaxDist: float, alpha: float) -> np.ndarray:
    r'''
    Given angle between shower direction (K) and line joining Xmax and observer's position,
    horizontal direction to observer's position, Xmax position and groundAltitude, compute
    coordinates of observer
    r'''

    # Compute rotation axis. Make sure it is normalized
    Rot_axis = np.cross(U,K)
    Rot_axis /= np.linalg.norm(Rot_axis)
    # Compute rotation matrix from Rodrigues formula
    Rotmat = rotation(-omega,Rot_axis)
    # Define rotation using scipy's method
    # Rotation = R.from_rotvec(-omega * Rot_axis)
    # print('#####')
    # print(Rotation.as_matrix())
    # print('#####')
    # Dir_obs  = Rotation.apply(K)
    Dir_obs = np.dot(Rotmat,K)
    # Compute observer's position
    # this assumed coincidence was computed at antenna altitude)
    # t = (Xant[2] - Xmax[2])/Dir_obs[2]
    # This assumes coincidence is computed at fixed alpha, i.e. along U, starting from Xcore
    t = np.sin(alpha)/np.sin(alpha+omega) * xmaxDist

    return Xmax + t*Dir_obs

@njit(**kwd)
def minor_equation(omega: float, n2: float, n1: float, alpha: float, delta: float, xmaxDist: float) -> float:
    ''' Compute time delay (in m) squared between two antennas separated by delta'''
    Lx = xmaxDist
    sa = np.sin(alpha)
    saw = np.sin(alpha+omega)
    com = np.cos(omega)
    l0 = Lx*sa/saw
    l1 = np.sqrt(l0**2+delta**2+2*delta*l0*com)
    l2 = np.sqrt(l0**2+delta**2-2*delta*l0*com)

    # Eq. 3.38 p125.
    res = (n2*l2+2*delta)**2-(n1*l1)**2
    
    return(res)

@njit(**kwd)
def master_equation(omega: float, n2: float, n1: float, alpha: float, delta: float, xmaxDist: float) -> float:
    '''Compute [c*delta(t)]^2 between two antennas separated by delta'''
    Lx = xmaxDist
    sa = np.sin(alpha)
    #saw = np.sin(alpha-omega) # Keeping minus sign to compare to Valentin's results. Should be plus sign.
    saw = np.sin(alpha+omega)
    com = np.cos(omega)
    l0 = Lx*sa/saw
    l1 = np.sqrt(l0**2+delta**2+2*delta*l0*com)
    l2 = np.sqrt(l0**2+delta**2-2*delta*l0*com)
    # Eq. 3.38 p125.
    res = (n2*l2-n1*l1+2*delta)**2
    return(res)

# Loss functions (chi2), according to different models:
# PWF: Plane wave function
# SWF: Spherical wave function
# ADF: Amplitude Distribution Function (see Valentin Decoene's thesis)

# ------------------------ PWF functions ------------------------

# @njit(**kwd)
def PWF_minimize_alternate_loss(Xants: np.ndarray, tants: np.ndarray, verbose: bool=False) -> np.ndarray:
    r'''Solves the minimization problem by using a special solution to the linear regression
    on K(\theta, \phi), with the ||K||=1 constraint. Note that this is a non-convex problem.
    This is formulated as 
    argmin_k k^T.A.k - 2 b^T.k, s.t. ||k||=1
    '''

    nants = tants.shape[0]

    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible : ", tants.shape, ' ≠ ', Xants.shape)
        return None
    
    # Compute A matrix (3x3) and b (3-)vector, see above
    PXT = Xants - Xants.mean(axis=0)  # P is the centering projector, XT=Xants
    A = np.dot(Xants.T, PXT)
    b = np.dot(Xants.T, tants-tants.mean(axis=0)) 
    # Diagonalize A, compute projections of b onto eigenvectors
    d, W = np.linalg.eigh(A)
    beta = np.dot(b, W)
    nbeta = np.linalg.norm(beta)

    if (np.abs(beta[0]/nbeta) < 1e-14):
        if (verbose):
            print("Degenerate case")
        # Degenerate case. This will be triggered e.g. when all antennas lie in a single plane.
        mu = -d[0]
        c = np.zeros(3, dtype=np.float64)
        c[1] = beta[1]/(d[1]+mu)
        c[2] = beta[2]/(d[2]+mu)
        si = np.sign(np.dot(W[:, 0], np.array([0, 0, 1.], dtype=np.float64)))
        c[0] = -si*np.sqrt(1-c[1]**2-c[2]**2)  # Determined up to a sign: choose descending solution
        k_opt = np.dot(W, c)
        # k_opt[2] = -np.abs(k_opt[2]) # Descending solution

    else:
        # Assume non-degenerate case, i.e. projections on smallest eigenvalue are non zero
        # Compute \mu such that \sum_i \beta_i^2/(\lambda_i+\mu)^2 = 1, using root finding on mu
        def nc(mu):
            # Computes difference of norm of k solution to 1. Coordinates of k are \beta_i/(d_i+\mu) in W basis
            c = beta/(d+mu)
            return ((c**2).sum()-1.)
        mu_min = -d[0]+beta[0]
        mu_max = -d[0]+np.linalg.norm(beta)
        mu_opt = brentq(nc, mu_min, mu_max, maxiter=1000)
        # Compute coordinates of k in W basis, return k
        c = beta/(d+mu_opt)
        k_opt = np.dot(W, c)

    # Now get angles from k_opt coordinates
    if k_opt[2] > 1e-2:
        k_opt = k_opt-2*(k_opt@W[:, 0])*W[:, 0]

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1], -k_opt[0])

    if phi_opt < 0:
        phi_opt += 2*np.pi

    return (np.array([theta_opt, phi_opt], dtype=np.float64))

# @njit(**kwd)
def PWF_minimize_alternate_loss_norm(Xants: np.ndarray, tants: np.ndarray, verbose: bool=False) -> np.ndarray:
    r'''Solves the minimization problem by using a special solution to the linear regression
    on K(\theta, \phi), with the ||K||=1 constraint. Note that this is a non-convex problem.
    This is formulated as 
    argmin_k k^T.A.k - 2 b^T.k, s.t. ||k||=1
    '''

    nants = tants.shape[0]

    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible : ", tants.shape, ' ≠ ', Xants.shape)
        return None
    
    # Compute A matrix (3x3) and b (3-)vector, see above
    sigma_tants = 5
    sigma_matrix = np.eye(nants) / sigma_tants**2
    wheigts_sum = sigma_matrix.diagonal().sum()
    Xants_weighted = (Xants.T @ sigma_matrix.diagonal()) / wheigts_sum
    tants_weighted = (tants @ sigma_matrix.diagonal()) / wheigts_sum
    PXT = Xants - Xants_weighted  # P is the centering projector, XT=Xants
    A = Xants.T @ sigma_matrix @ PXT
    b = Xants.T @ sigma_matrix @ (tants-tants_weighted)
    # Diagonalize A, compute projections of b onto eigenvectors
    d, W = np.linalg.eigh(A)
    beta = np.dot(b, W)
    nbeta = np.linalg.norm(beta)

    if (np.abs(beta[0]/nbeta) < 1e-14):
        # if (verbose):
        #     print("Degenerate case")
        # Degenerate case. This will be triggered e.g. when all antennas lie in a single plane.
        mu = -d[0]
        c = np.zeros(3, dtype=np.float64)
        c[1] = beta[1]/(d[1]+mu)
        c[2] = beta[2]/(d[2]+mu)
        si = np.sign(np.dot(W[:, 0], np.array([0, 0, 1.], dtype=np.float64)))
        c[0] = -si*np.sqrt(1-c[1]**2-c[2]**2)  # Determined up to a sign: choose descending solution
        k_opt = np.dot(W, c)
        # k_opt[2] = -np.abs(k_opt[2]) # Descending solution

    else:
        # Assume non-degenerate case, i.e. projections on smallest eigenvalue are non zero
        # Compute \mu such that \sum_i \beta_i^2/(\lambda_i+\mu)^2 = 1, using root finding on mu
        def nc(mu):
            # Computes difference of norm of k solution to 1. Coordinates of k are \beta_i/(d_i+\mu) in W basis
            c = beta/(d+mu)
            return ((c**2).sum()-1.)
        mu_min = -d[0]+beta[0]
        mu_max = -d[0]+np.linalg.norm(beta)
        mu_opt = brentq(nc, mu_min, mu_max, maxiter=1000)
        # Compute coordinates of k in W basis, return k
        c = beta/(d+mu_opt)
        k_opt = np.dot(W, c)

    # Now get angles from k_opt coordinates
    if k_opt[2] > 1e-2:
        k_opt = k_opt-2*(k_opt@W[:, 0])*W[:, 0]

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1], -k_opt[0])

    if phi_opt < 0:
        phi_opt += 2*np.pi
    
    if verbose and 3<2: # Change to 1<2 to enable verbose output
        mask = (tants != 0)
        n_valid = mask.sum()

        tants_centered = tants[mask] - tants[mask].mean()
        Xants_centered = Xants[mask] - Xants[mask].mean(axis=0)

        predictions = np.dot(Xants_centered, k_opt)
        residuals = tants_centered - predictions

        residuals_rms = np.sqrt(np.mean(residuals**2))
        chi2 = np.sum((residuals / sigma_tants)**2)
        ndof = n_valid - 3  # θ, φ, t0
        chi2_reduced = chi2 / ndof

        print(f"Nombre d'antennes valides : {n_valid}")
        print(f"χ² = {chi2}")
        print(f"χ² réduit = {chi2_reduced} (attendu ≈ 1 si σ={sigma_tants}ns)")
        print(f"Résidus RMS = {residuals_rms} ns")
        print(f"Jitter effectif mesuré = {residuals_rms} ns")

    return (np.array([theta_opt, phi_opt], dtype=np.float64))

@njit(**kwd)
def PWF_loss(params: np.ndarray, Xants: np.ndarray, tants: np.ndarray, verbose: bool=False) -> float:

    r'''
    Defines Chi2 by summing model residuals
    over antenna pairs (i, j):
    loss = \sum_{i>j} ((Xants[i, :]-Xants[j, :]).K - cr(tants[i]-tants[j]))**2
    where:
    params=(theta, phi): spherical coordinates of unit shower direction vector K
    Xants are the antenna positions (shape=(nants, 3))
    tants are the antenna arrival times of the wavefront (trigger time, shape=(nants, ))
    cr is radiation speed, by default 1 since time is expressed in m.
    r'''

    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp, st*sp, ct], dtype=np.float64)
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Use numpy outer methods to build matrix X_ij = x_i -x_j
    xk = np.dot(Xants, K)
    DXK = np.subtract.outer(xk, xk)
    DT  = np.subtract.outer(tants, tants)
    chi2 = ( (DXK - pr.cr*DT)**2 ).sum() / 2. # Sum over upper triangle, diagonal is zero because of antisymmetry of DXK, DT
    if verbose:
        print("params = ", params*180./np.pi)
        print("Chi2 = ", chi2)
    return(chi2)

@njit(**kwd)
def PWF_alternate_loss(params: np.ndarray, Xants: np.ndarray, tants: np.ndarray, verbose=False) -> float:
    r'''
    Defines Chi2 by summing model residuals over individual antennas, 
    after maximizing likelihood over reference time.
    r'''
    nants = tants.shape[0]
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Make sure tants and Xants are compatible
    residuals = PWF_residuals(params, Xants, tants, verbose=verbose)
    chi2 = (residuals**2).sum()
    return(chi2)

@njit(**kwd)
def PWF_residuals(params: np.ndarray, Xants: np.ndarray, tants: np.ndarray) -> float:

    r'''
    Computes timing residuals for each antenna using plane wave model
    Note that this is defined at up to an additive constant, that when minimizing
    the loss over it, amounts to centering the residuals.
    r'''
    nants = tants.shape[0]
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None

    times = PWF_model(params, Xants)
    res = pr.cr * (tants - times)
    res -= res.mean()  # Mean is projected out when maximizing likelihood over reference time t0
    return (res)

@njit(**kwd)
def PWF_simulation(params: np.ndarray, Xants: np.ndarray, iseed=None) -> np.ndarray:
    r'''
    Generates plane wavefront timings, zero at shower core, with jitter noise added
    r'''

    times = PWF_model(params,Xants)
    # Add noise
    if (iseed is not None):
        np.random.seed(iseed)
    n = np.random.standard_normal(times.size) * pr.sigma_t * pr.c_light
    return (times + n)

@njit(**kwd)
def PWF_model(params: np.ndarray, Xants: np.ndarray) -> np.ndarray:
    r'''
    Generates plane wavefront timings
    r'''
    theta, phi = params
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp=np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct], dtype=np.float64)
    dX = Xants - np.array([0.,0.,pr.groundAltitude], dtype=np.float64)
    tants = np.dot(dX,K) / pr.cr 
 
    return (tants)

# ------------------------ SWF functions ------------------------

@njit(**kwd)
def SWF_model(params: np.ndarray, Xants: np.ndarray) -> np.ndarray:
    r"""Computes predicted wavefront timings for the spherical case.

    Parameters
    ----------
    Xants : array (nants, 3)
        Antenna positions.
    tants : array (nants,)
        Trigger times.
    params : (alpha, beta, r_xmax, t_s)
        Source parameters: direction (alpha, beta), distance r_xmax,
        and emission time t_s.
    cr : float
        Propagation speed in the medium (default 1, time in meters).

    Outputs
    -------
    loss : float
        Value of the loss with given parameters"""
    # Unpack parameters
    alpha, beta, r_xmax, t_s = params
    nants = Xants.shape[0]

    # Build K and Xmax
    ca = np.cos(alpha); sa = np.sin(alpha); cb = np.cos(beta); sb = np.sin(beta)
    K = np.array([-sa*cb, -sa*sb, -ca], dtype=np.float64)
    Xmax = -r_xmax * K + np.array([0., 0., pr.groundAltitude], dtype=np.float64)

    tants = np.zeros(nants, dtype=np.float64)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i, :])
        dX = Xants[i, :] - Xmax
        tants[i] = t_s + n_average / pr.cr * np.linalg.norm(dX)

    return (tants)

@njit(**kwd)
def SWF_loss(params: np.ndarray, Xants: np.ndarray, tants: np.ndarray, ndof: bool=False):
    """Compute the chi-square loss as the sum of squared timing residuals over antennas.

    For each antenna i, the residual is defined as the difference between:
    - the measured trigger time corrected by the source emission time, and
    - the geometric propagation distance from the source to the antenna.

    loss = sum_i [ cr * (tants[i] - t_s)
                - sqrt((Xants[i,0] - x_s)^2
                        + (Xants[i,1] - y_s)^2
                        + (Xants[i,2] - z_s)^2) ]^2

    Parameters
    ----------
    Xants : array (nants, 3)
        Antenna positions.
    tants : array (nants,)
        Trigger times.
    params : (alpha, beta, r_xmax, t_s)
        Source parameters: direction (alpha, beta), distance r_xmax,
        and emission time t_s.
    cr : float
        Propagation speed in the medium (default 1, time in meters).

    Outputs
    -------
    loss : float
        Value of the loss with given parameters"""
    
    alpha, beta, r_xmax, t_s = params
    nants = tants.shape[0]

    # Calcul de K
    ca = np.cos(alpha); sa = np.sin(alpha); cb = np.cos(beta); sb = np.sin(beta)
    K = np.array([-sa*cb,-sa*sb,-ca], dtype=np.float64) # minus direction of the Xmax position vector
    sigma_t = pr.jitter_time * pr.c_light  # Convert ns to meters

    # Calcul de la position de X_max
    Xmax = -r_xmax * K + np.array([0.,0.,pr.groundAltitude], dtype=np.float64) # Xmax is in the opposite direction to K
    
    tmp = 0. # Initialize chi2
    for i in range(nants):
        # Compute average refraction index between emission and observer
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        dX = Xants[i,:] - Xmax # Vector between Xmax and antenna i

        # Spherical wave front chi2 calculation
        res = pr.cr*(tants[i]-t_s) - n_average*np.linalg.norm(dX)
        tmp += (res/sigma_t) **2

    if ndof:
        chi2_ndof = tmp / (nants - 4)  # 4 parameters: alpha, beta, r_xmax, t_s
        return chi2_ndof
    else:
        chi2 = tmp
        return chi2

@njit(**kwd)
def SWF_residuals(params: np.ndarray, Xants: np.ndarray, tants: np.ndarray) -> np.ndarray:

    r'''
    Computes timing residuals for each antenna (i):
    residual[i] = ( cr(tants[i]-t_s) - \sqrt{(Xants[i,0]-x_s)**2)+(Xants[i,1]-y_s)**2+(Xants[i,2]-z_s)**2} )**2
    where:
    Xants are the antenna positions (shape=(nants,3))
    tants are the trigger times (shape=(nants,))
    x_s = \sin(\theta)\cos(\phi)
    y_s = \sin(\theta)\sin(\phi)
    z_s = \cos(\theta)

    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical coordinates of the vector K
    t_s is the source emission time
    cr is the radiation speed in medium, by default 1 since time is expressed in m.
    r'''

    alpha, beta, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ca = np.cos(alpha); sa = np.sin(alpha); cb = np.cos(beta); sb = np.sin(beta)
    K = np.array([-sa*cb,-sa*sb,-ca], dtype=np.float64)
    Xmax = -r_xmax * K + np.array([0.,0.,pr.groundAltitude], dtype=np.float64) # Xmax is in the opposite direction to shower propagation.

    res = np.zeros(nants, dtype=np.float64)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        dX = Xants[i,:] - Xmax
        res[i] = pr.cr*(tants[i]-t_s) - n_average*np.linalg.norm(dX)
    return(res)

@njit(**kwd)
def SWF_simulation(params: np.ndarray, Xants: np.ndarray, iseed=1234) -> np.ndarray:
    r'''
    Computes simulated wavefront timings for the spherical case.
    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical angular coordinates of Xmax, and  
    r_xmax is the distance of Xmax to the reference point of coordinates (0,0,groundAltitude)
    sigma_t is the timing jitter noise, in ns
    iseed is the integer random seed of the noise generator
    c_r is the speed of light in vacuum, in units of c_light
    r'''
    theta, phi, r_xmax, t_s = params
    nants = Xants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([-st*cp, -st*sp, -ct], dtype=np.float64)
    Xmax = -r_xmax * K + np.array([0.,0.,pr.groundAltitude], dtype=np.float64)
    tants = np.zeros(nants, dtype=np.float64)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        dX = Xants[i,:] - Xmax
        tants[i] = t_s + n_average / pr.cr * np.linalg.norm(dX)

    np.random.seed(iseed)
    n = np.random.standard_normal(tants.size) * pr.jitter_time * pr.c_light
    return (tants + n)

# ------------------------ ADF functions ------------------------

@njit(**kwd)
def ADF_3D_parameters(params: np.ndarray, Aants: np.ndarray, Xants: np.ndarray, Xmax: np.ndarray):
    
    r'''

    Computes amplitude prediction for each antenna (i):
    residuals[i] = f_i^{ADF}(\theta,\phi,\delta\omega,A,r_xmax)
    where the ADF function reads:
    
    f_i = f_i(\omega_i, \eta_i, \alpha, l_i, \delta_omega, A)
        = A/l_i f_geom(\alpha, \eta_i) f_Cerenkov(\omega,\delta_\omega)
    
    where 
    
    f_geom(\alpha, \eta_i) = (1 + B \sin(\alpha))**2 \cos(\eta_i) # B is here the geomagnetic asymmetry
    f_Cerenkov(\omega_i,\delta_\omega) = 1 / (1+4{ (\tan(\omega_i)/\tan(\omega_c))**2 - 1 ) / \delta_\omega }**2 )
    
    Input parameters are: params = theta, phi, delta_omega, amplitude
    \theta, \phi define the shower direction angles, \delta_\omega the width of the Cerenkov ring, 
    A is the amplitude paramater, r_xmax is the norm of the position vector at Xmax.

    Derived parameters are: 
    \alpha, angle between the shower axis and the magnetic field
    \eta_i is the azimuthal angle of the (projection of the) antenna position in shower plane
    \omega_i is the angle between the shower axis and the vector going from Xmax to the antenna position

    r'''

    # Basic parameters
    theta, phi, delta_omega, amplitude = params
    nants = Xants.shape[0]
    Bvec = pr.B_vec
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)

    # Define shower basis vectors
    K = np.array([-st*cp,-st*sp,-ct], dtype=np.float64)
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
   
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K))
     
    # Calcul de la distance Xmax - ground, c'est pas juste r_xmax ???
    XmaxDist = (pr.groundAltitude-Xmax[2])/K[2]
    theta_deg = theta * 180.0 / np.pi

    # Calcul de f_geom
    asym_coeff = -0.003*theta_deg+0.220 # Calcul de G_A
    asym = asym_coeff/np.sqrt(1. - np.dot(K,Bvec)**2) # Calcul de G_A/sin(alpha) où alpha = |KxB|


    # Boucle sur les antennes — pas de table de pré-calcul possible pour l’angle de Tcherenkov
    # Le calcul doit être fait pour chaque antenne.
    res = np.zeros(nants, dtype=np.float64)
    eta_array = np.zeros(nants, dtype=np.float64)
    omega_array = np.zeros(nants, dtype=np.float64)
    omega_cr_analytic_array = np.zeros(nants, dtype=np.float64)
    omega_cr_analytic_effectif_array = np.zeros(nants, dtype=np.float64)
    omega_cerenkov_simu_array = np.zeros(nants, dtype=np.float64)

    Xa = Xmax + 2.0e3 * K
    Xb = Xmax - 2.0e3 * K

    for i in range(nants):
        # Position de l’antenne par rapport à Xmax
        dX    = Xants[i, :] - Xmax
        dX_sp = np.dot(mat, dX)  # Coordonnées dans le repère du shower

        l_ant = np.linalg.norm(dX)
        eta   = np.arctan2(dX_sp[1], dX_sp[0])
        omega = np.arccos(np.dot(K, dX) / l_ant)

        # Angles de Tcherenkov (simulation et formules analytiques)
        omega_cr              = compute_Cerenkov_3D(Xants[i, :], K, XmaxDist, Xmax, 2.0e3)
        omega_cr_analytic     = np.arccos(1.0 / RefractionIndexAtPosition(Xmax))
        omega_cr_analytic_eff = np.arccos(1.0 / ZHSEffectiveRefractionIndex(Xmax, np.array([0, 0, pr.groundAltitude], dtype=np.float64)))

        # Limitation de l’angle pour petits angles d’incidence
        if theta_deg < 70:
            omega_cr = min(omega_cr, np.deg2rad(0.6))

        # Distribution d’amplitude
        ratio = (np.tan(omega) / np.tan(omega_cr)) ** 2 - 1
        adf   = amplitude / l_ant / (1.0 + 4.0 * (ratio / delta_omega) ** 2)
        adf  *= 1.0 + asym * np.cos(eta)

        # Résidu (Chi2)
        res[i] = Aants[i] - adf

        # Sauvegarde des valeurs
        eta_array[i]                        = eta
        omega_array[i]                      = omega
        omega_cr_analytic_array[i]          = omega_cr_analytic
        omega_cr_analytic_effectif_array[i] = omega_cr_analytic_eff
        omega_cerenkov_simu_array[i]        = omega_cr

    return (
        eta_array,
        omega_array,
        omega_cerenkov_simu_array,
        omega_cr_analytic_array,
        omega_cr_analytic_effectif_array,
    )

@njit(**kwd)
def ADF_loss(params: np.ndarray, Aants: np.ndarray, Xants: np.ndarray, Xmax: np.ndarray, ndof: bool=False):
    '''Compute chi2 between measured amplitudes and 3D ADF model. Cleaned version
    
    Inputs
    ------
    params : tuple (theta, phi, delta_omega, amplitude)
        theta : float - Zenith angle in radians (0=vertical, π/2=horizontal), this angle is define at Xmax
        phi : float - Azimuthal angle in radians, defined at Xmax too
        delta_omega : float - Angular width of emission distribution (radians)
        amplitude : float - Overall signal amplitude
    Aants : ndarray (nants,)
        Measured amplitudes at each antenna.
    Xants : ndarray (nants, 3)
        Antenna positions.
    Xmax : ndarray (3,)
        Position of shower maximum.
    ndof : bool
        If True, return reduced chi2 (divided by number of degrees of freedom). Default is False.
    asym_coeff : float
        Coefficient for geomagnetic asymmetry. Default is 0.01.
        
    Outputs
    -------
    chi2 : float
        Chi2 value between measured amplitudes and model predictions.'''

    # Basic parameters
    theta, phi, delta_omega, amplitude = params
    nants = Xants.shape[0]
    Bvec = pr.B_vec
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)

    # Define shower basis vectors
    K = np.array([-st*cp,-st*sp,-ct], dtype=np.float64)
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
   
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K)).astype(np.float64)

    # Calculation of Xmax - ground distance
    XmaxDist = np.linalg.norm(np.array([0, 0, pr.groundAltitude], dtype=np.float64) - Xmax) 
    theta_deg = theta * 180.0 / np.pi

    # Calculation of f_geom
    asym_coeff = -0.003*theta_deg+0.220 # Calcul de G_A
    asym = asym_coeff/np.sqrt(1. - np.dot(K,Bvec)**2) # Calcul de G_A/sin(alpha) où alpha = |KxB|

    # Loop on antennas. Here no precomputation table is possible for Cerenkov angle computation.
    # Calculation needs to be done for each antenna.
    tmp = 0. # Initialize chi2
    uncertainties = ((0.075 * np.abs(Aants))**2 + pr.galactic_noise_floor**2)**0.5

    for i in range(nants):
        
        dX = Xants[i,:]-Xmax
        dX_sp = np.dot(mat,dX)
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1],dX_sp[0])
        omega = np.arccos(np.dot(K,dX)/l_ant)
        
        omega_cr = compute_Cerenkov_3D_2(Xants[i,:],K,XmaxDist,Xmax,2.0e3)

        if theta_deg < 70: 
            omega_cr = min(omega_cr, np.deg2rad(0.6))

        adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega)**2)
        adf *= 1. + asym*np.cos(eta)

        # Robust uncertainty calculation (7.5%) + 8mv/m galactic noise floor
        tmp += (Aants[i]-adf)**2 / uncertainties[i]**2

    if ndof:
        chi2_ndof = tmp / (nants - 4)  # 4 parameters: theta, phi, delta_omega, amplitude
        return chi2_ndof
    else:
        chi2 = tmp
        return chi2

@njit(**kwd) #{"fastmath": {"reassoc", "contract", "arcp"}}
def ADF_3D_model(params: np.ndarray, Xants: np.ndarray, Xmax: np.ndarray) -> np.ndarray:
    
    '''
    Calculate radio signal received by each antenna from an atmospheric shower.
    
    Models the radio emission from a particle shower propagating 
    through the atmosphere. Computes the signal at each antenna based on Cherenkov 
    emission geometry and magnetic field effects.
    
    Parameters
    ----------
    params : tuple (theta, phi, delta_omega, amplitude)
        theta : float - Zenith angle in radians (0=vertical, π/2=horizontal)
        phi : float - Azimuthal angle in radians
        delta_omega : float - Angular width of emission distribution (radians)
        amplitude : float - Overall signal amplitude
    Xants : ndarray (nants, 3)
        Antenna positions [x, y, z] in meters
    Xmax : ndarray (3,)
        Shower maximum position [x, y, z] in meters
    Bvec : ndarray (3,)
        Earth's magnetic field vector
    groundAltitude : float
        Ground altitude in meters
        
    Returns
    -------
    ndarray (nants,)
        Signal amplitude at each antenna. Scales as 1/distance with angular 
        distribution peaked at Cherenkov angle and azimuthal asymmetry from 
        magnetic field.
    '''

    # --- Unpack parameters ---
    theta, phi, delta_omega, amplitude = params
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)

    # --- Define shower axis and orthogonal basis ---
    K = np.array([-st * cp, -st * sp, -ct], dtype=np.float64)
    Bvec = pr.B_vec
    # Build orthonormal frame (KxB, KxKxB, K)
    KxB = np.cross(K, Bvec)
    KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K, KxB)
    KxKxB /= np.linalg.norm(KxKxB)
    mat = np.vstack((KxB, KxKxB, K))  # rotation matrix to shower frame

    # --- Geometry-related quantities ---
    XmaxDist = np.linalg.norm(np.array([0, 0, pr.groundAltitude], dtype=np.float64) - Xmax) 
    theta_deg = np.degrees(theta)
    # Empirical asymmetry term (depends on geomagnetic angle)
    asym = (-0.003 * theta_deg + 0.220) / np.sqrt(1. - np.dot(K, Bvec)**2)

    # --- Loop over antennas ---
    res = np.empty(len(Xants))
    for i, ants in enumerate(Xants):
        # Vector from Xmax to antenna
        dX = ants - Xmax
        # Expressed in shower coordinate frame
        dX_sp = mat @ dX
        l = np.linalg.norm(dX)  # antenna distance from Xmax

        # Angular coordinates in shower frame
        eta = np.arctan2(dX_sp[1], dX_sp[0])
        omega = np.arccos(np.dot(K, dX) / l)

        # Compute local Cherenkov angle (depends on ground altitude)
        omega_cr = compute_Cerenkov_3D_2(ants, K, XmaxDist, Xmax, 2e3)

        if theta_deg < 70:
            # Empirical upper bound for shallow showers
            omega_cr = min(omega_cr, np.radians(0.6))

        # --- Analytical distribution function (ADF) ---
        ratio = (np.tan(omega) / np.tan(omega_cr))**2 - 1
        adf = amplitude / l / (1 + 4 * (ratio / delta_omega)**2)
        adf *= 1 + asym * np.cos(eta)  # add asymmetry in azimuth

        res[i] = adf

    return res

# ------------------------ Cerenkov functions ------------------------

@njit(**kwd)
def compute_alpha_3D(Xant: np.ndarray, K: np.ndarray) -> float:
    dXcore = Xant - np.array([0.,0.,pr.groundAltitude], dtype=np.float64) 
    U = dXcore / np.linalg.norm(dXcore)
    # Compute angle between shower direction and (horizontal) direction to observer
    alpha = np.arccos(np.dot(K,U))
    alpha = np.pi-alpha
    return (alpha)

@njit(**kwd)
def compute_U(Xant: np.ndarray) -> np.ndarray:
    dXcore = Xant - np.array([0.,0.,pr.groundAltitude], dtype=np.float64) 
    U = dXcore / np.linalg.norm(dXcore)
    return (U)

@njit(**kwd)
def compute_observer_position_3D(omega: float, Xmax: np.ndarray, U: np.ndarray, K: np.ndarray, xmaxDist: float, alpha: float) -> np.ndarray:
    r'''
    Given angle omega between shower direction (K) and line joining Xmax and observer's position,
    Xmax position and Xant antenna position, and unit vector (U) to observer from shower core, compute
    coordinates of observer
    r'''

    # Compute rotation axis. Make sure it is normalized
    Rot_axis = np.cross(U,K)
    Rot_axis /= np.linalg.norm(Rot_axis)
    # Compute rotation matrix from Rodrigues formula
    Rotmat = rotation(-omega,Rot_axis)
    # Define rotation using scipy's method
    # Rotation = R.from_rotvec(-omega * Rot_axis)
    # print('#####')
    # print(Rotation.as_matrix())
    # print('#####')
    # Dir_obs  = Rotation.apply(K)
    Dir_obs = np.dot(Rotmat,K)
    # Compute observer's position
    # this assumed coincidence was computed at antenna altitude)
    #t = (Xant[2] - Xmax[2])/Dir_obs[2]
    # This assumes coincidence is computed at fixed alpha, i.e. along U, starting from Xcore
    t = np.sin(alpha)/np.sin(alpha+omega) * xmaxDist
    X = Xmax + t*Dir_obs
    return (X)

@njit(**kwd)
def compute_delay_3D(omega: float, Xmax: np.ndarray, Xa: np.ndarray, Xb: np.ndarray, U: np.ndarray, K: np.ndarray, alpha: float, delta: float, xmaxDist: float) -> float:

    X = compute_observer_position_3D(omega,Xmax,U,K,xmaxDist,alpha)
    # print('omega = ',omega,'X_obs = ',X)
    n2 = ZHSEffectiveRefractionIndex(Xa,X)
    # print('n0 = ',n0)
    n1 = ZHSEffectiveRefractionIndex(Xb,X)
    # print('n1 = ',n1)
    res = minor_equation(omega,n2,n1,alpha, delta, xmaxDist)
    # print('delay = ',res)
    return(res)

@njit(**kwd)
def compute_delay_3D_master_equation(omega: float, Xmax: np.ndarray, Xa: np.ndarray, Xb: np.ndarray, Xant: np.ndarray, U: np.ndarray, K: np.ndarray, alpha: float, delta: float, xmaxDist: float) -> float:

    X = compute_observer_position_3D(omega,Xmax,Xant,U,K,xmaxDist,alpha)
    # print('omega = ',omega,'X_obs = ',X)
    n2 = ZHSEffectiveRefractionIndex(Xa, X)
    # print('n0 = ',n0)
    n1 = ZHSEffectiveRefractionIndex(Xb, X)
    # print('n1 = ',n1)
    res = master_equation(omega, n2, n1, alpha, delta, xmaxDist)
    # print('delay = ',res)
    return(res)

@njit(**kwd)
def compute_Cerenkov_3D(Xant: np.ndarray, K: np.ndarray, xmaxDist: float, Xmax: np.ndarray, delta: float) -> float:

    r'''
    Solve for Cerenkov angle by minimizing
    time delay between light rays originating from Xb and Xmax and arriving
    at observer's position. 
    Xant:  (single) antenna position 
    K:     direction vector of shower
    Xmax:  coordinates of Xmax point
    delta: distance between Xmax and Xb points
    groundAltitude: self explanatory

    Returns:     
    omega: angle between shower direction and line joining Xmax and observer's position

    r'''

    # Compute coordinates of point before Xmax
    Xb = Xmax - delta*K
    # Compute coordinates of point after Xmax
    Xa = Xmax +delta*K

    #dXcore = Xant - np.array([0.,0.,groundAltitude])
    # Core of shower, taken at groundAltitude for reference
    # Ground altitude might be computed later as a derived quantity, e.g. 
    # as the median of antenna altitudes.
    Xcore = Xmax + xmaxDist * K 
    dXcore = Xant - Xcore

    # Direction vector to observer's position from shower core
    # This is a bit dangerous for antennas numerically close to shower core... 
    U = dXcore / np.linalg.norm(dXcore)
    # Compute angle between shower direction and (horizontal) direction to observer
    alpha = np.arccos(np.dot(K,U))
    alpha = np.pi-alpha


    # Now solve for omega
    # Starting point at standard value acos(1/n(Xmax)) 
    omega_cr_guess = np.arccos(1./RefractionIndexAtPosition(Xmax))
    # print("###############")
    # omega_cr = fsolve(compute_delay,[omega_cr_guess])
    omega_cr = newton(compute_delay_3D, omega_cr_guess, args=(Xmax, Xa, Xb, Xant,U,K,alpha,delta, xmaxDist),verbose=False)
    ### DEBUG ###
    # omega_cr = omega_cr_guess
    return(omega_cr)

@njit(**kwd)
def compute_Cerenkov_3D_2(Xant: np.ndarray, K: np.ndarray, XmaxDist: float, Xmax: np.ndarray, delta: float) -> float:
    """
    Compute the Cerenkov angle in 3D geometry by solving the master equation.
    
    :param Xant: Position of the antenna
    :type Xant: np.ndarray
    :param K: Direction vector of the shower, defined at Xmax
    :type K: np.ndarray
    :param XmaxDist: Distance from Xmax to the center of the array (0, 0, groundAltitude)
    :type XmaxDist: float
    :param Xmax: Position of the Xmax point
    :type Xmax: np.ndarray
    :param delta: Distance parameter for points before and after Xmax along the shower direction
    :type delta: float
    :return: Cerenkov angle
    :rtype: float
    """

    # Compute coordinates of points before and after Xmax
    Xa = Xmax + delta*K
    Xb = Xmax - delta*K

    # Compute core position at ground altitude
    theta = np.arccos(-K[2])
    phi = np.arctan2(-K[1], -K[0])
    z_core = Xant[2]
    y_core = np.sin(theta)*np.sin(phi)*(z_core-Xmax[2])/(-np.cos(theta)) + Xmax[1]
    x_core = np.sin(theta)*np.cos(phi)*(z_core-Xmax[2])/(-np.cos(theta)) + Xmax[0]
    Xcore = np.empty(3, dtype=np.float64)
    Xcore[0] = x_core
    Xcore[1] = y_core
    Xcore[2] = z_core

    # Compute vector from core to antenna
    dXcore = Xant - Xcore
    norm_dX = np.sqrt(np.sum(dXcore**2))
    U = dXcore / norm_dX

    # Compute angle alpha between shower direction and direction to observer
    alpha = np.arccos(min(max(np.dot(K, U), -1.0), 1.0))
    alpha = np.pi - alpha

    omega_cr_guess = np.arccos(1.0/RefractionIndexAtPosition(Xmax))
    omega_cr = newton(compute_delay_3D, omega_cr_guess, args=(Xmax, Xa, Xb, U, K, alpha, delta, XmaxDist), tol=1e-7)
    return omega_cr

@njit(**kwd)
def ADF_grad(params: np.ndarray, Aants: np.ndarray, Xants: np.ndarray, Xmax: np.ndarray) -> np.ndarray:
    
    theta, phi, delta_omega, amplitude = params
    nants = Aants.shape[0]
    Bvec = pr.B_vec
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([-st*cp,-st*sp,-ct], dtype=np.float64)
    K_plan = np.array([K[0],K[1]], dtype=np.float64)
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K))
    # 
    XmaxDist = (pr.groundAltitude-Xmax[2])/K[2]
    # print('XmaxDist = ',XmaxDist)
    asym = pr.assym_coeff * (1. - np.dot(K,Bvec)**2) # Azimuthal dependence, in \sin^2(\alpha)
    #
    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of Aants and Xants are incompatible",Aants.shape, Xants.shape)
        return None

    # Precompute an array of Cerenkov angles to interpolate over (as in Valentin's code)
    omega_cerenkov = np.zeros(2*pr.n_omega_cr+1, dtype=np.float64)
    xi_table = np.arange(2*pr.n_omega_cr+1)/pr.n_omega_cr*np.pi
    for i in range(pr.n_omega_cr+1):
        omega_cerenkov[i] = compute_Cerenkov_3D(xi_table[i],K,XmaxDist,Xmax,2.0e3)
    # Enforce symmetry
    omega_cerenkov[pr.n_omega_cr+1:] = (omega_cerenkov[:pr.n_omega_cr])[::-1]

    #Output antenas amplitudes for comparisons
    Aants_out = np.zeros((nants, 2), dtype=np.float64)
    # Loop on antennas
    jac = np.zeros(4, dtype=np.float64)
    for i in range(nants):
        # Antenna position from Xmax
        dX = Xants[i,:]-Xmax
        # Expressed in shower frame coordinates
        dX_sp = np.dot(mat,dX)
        #
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1],dX_sp[0])
        omega = np.arccos(np.dot(K,dX)/l_ant)

        omega_cr = compute_Cerenkov_3D(Xants[i,:],K,XmaxDist,Xmax,2.0e3)
        width = ct / (dX[2]/l_ant) * delta_omega
        # Distribution
        f_cerenkov = 1. / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/width )**2)
        adf = amplitude/l_ant * f_cerenkov
        adf *= 1. + asym*np.cos(eta) #
        res = Aants[i] - adf
        #
        dK_dtheta = np.array([ct*cp, ct*sp,-st], dtype=np.float64)
        dfgeom_dtheta = -2.*pr.asym_coeff*np.cos(eta)*(np.dot(K,Bvec))*(np.dot(dK_dtheta,Bvec))
        dres_dtheta = (-amplitude/l_ant)*f_cerenkov*dfgeom_dtheta
        #
        dK_dphi = np.array([-st*sp, st*cp, 0.], dtype=np.float64)
        dfgeom_dphi = -2.*pr.asym_coeff*np.cos(eta)*(np.dot(K,Bvec))*(np.dot(dK_dphi,Bvec))
        dres_dphi = (-amplitude/l_ant)*f_cerenkov*dfgeom_dphi
        #
        term1 = (np.tan(omega)/np.tan(omega_cr))**2 - 1. 
        dfcerenkov_ddelta_omega = (8.*l_ant**2.*(1/ct**2.)*term1**2.)/(delta_omega**3.*dX[2]**2.*(1+(4.*l_ant**2.*(1/ct**2.)*term1**2.)/(delta_omega**2.*dX[2]**2.))**2.)
        dres_ddelta_omega =  (-amplitude/l_ant)*(1. + asym*np.cos(eta))*dfcerenkov_ddelta_omega
        #
        dres_damplitude = (-1./l_ant) * f_cerenkov * (1. + asym*np.cos(eta))
        # grad
        jac[0] += 2.*res*dres_dtheta
        jac[1] += 2.*res*dres_dphi
        jac[2] += 2.*res*dres_ddelta_omega
        jac[3] += 2.*res*dres_damplitude

        return(jac)
