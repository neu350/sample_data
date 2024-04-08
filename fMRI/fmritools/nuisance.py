import numpy as np
from scipy.special import legendre as _legendre

def legendre(t, order=0):
    """Generate Legendre polynomial regressors.
    
    Parameters
    ----------
    n_acq : int
        Number of time points.
    order : int
        Highest order of Legendre polynimals to generate.
        
    Returns
    -------
    Z : array, shape=(t,order+1)
        Legendre polynomial regressors.
    """
    
    ## Error catching.
    assert order >= 0
    
    ## Define grid.
    x = np.linspace(-1,1,t)
    
    ## Return Legendre polynomials.
    return np.column_stack([_legendre(i)(x) for i in range(order+1)])