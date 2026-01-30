import numpy as np
from numba import njit

@njit
def der(func,x,args=[], eps=1e-7):
    '''
    Forward estimate of derivative
    '''
    return ((func(x+eps,*args)-func(x,*args))/eps)

@njit
def newton(func, x0, tol=1e-7, nstep_max=100, args=[], verbose=False):
    '''
    Newton method for zero finding.
    Uses forward estimate of derivative.
    Robust version with safeguards against division by zero.
    '''
    rel_error = np.infty
    xold = x0
    nstep = 0
    
    # Protection contre x0 = 0
    if abs(xold) < 1e-15:
        xold = 1e-10  # Petite valeur non nulle
    
    while ((rel_error > tol) and (nstep < nstep_max)):
        f_val = func(xold, *args)
        df_val = der(func, xold, args=args)
        
        # Protection contre dérivée nulle
        if abs(df_val) < 1e-15:
            if verbose:
                print(f"WARNING: derivative near zero at iteration {nstep}, df = {df_val}")
            # Essayer une petite perturbation
            xold += 1e-8
            df_val = der(func, xold, args=args)
            if abs(df_val) < 1e-15:
                if verbose:
                    print("Newton failed: derivative is zero")
                return xold  # Retourner la meilleure estimation
        
        # Protection contre fonction nulle (déjà à la solution)
        if abs(f_val) < 1e-15:
            if verbose:
                print(f"Solution found at iteration {nstep}: f(x) ≈ 0")
            return xold
        
        # Mise à jour Newton
        delta = f_val / df_val
        x = xold - delta
        nstep += 1
        
        if verbose:
            print(f"Iteration {nstep}: x = {x}, f(x) = {f_val}, df(x) = {df_val}")
        
        # Calcul de l'erreur relative robuste
        if abs(xold) > 1e-10:
            rel_error = abs((x - xold) / xold)
        else:
            # Si xold très petit, utiliser erreur absolue
            rel_error = abs(x - xold)
        
        # Protection contre divergence
        if abs(x) > 1e10:
            if verbose:
                print(f"WARNING: Newton diverging at iteration {nstep}, |x| = {abs(x)}")
            return xold  # Retourner la valeur précédente
        
        # Protection contre NaN
        if not np.isfinite(x):
            if verbose:
                print(f"WARNING: Newton produced NaN/Inf at iteration {nstep}")
            return xold
        
        xold = x
    
    if nstep == nstep_max and verbose:
        print(f"WARNING: Convergence not achieved in {nstep_max} iterations")
    
    return x


def sqr(x):
    return (x**2-1)

def main():
    newton(sqr,4.0,verbose=True)

if __name__ == '__main__':
    main()

