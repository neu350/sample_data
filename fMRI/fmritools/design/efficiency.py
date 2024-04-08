import numpy as np
from ..hrf import spm_hrf

def _put(a, ind, v, mode='raise'):
    """np.put, not in-place."""
    arr = a.copy()
    np.put(arr, ind, v, mode)
    return arr

def detection_power(X, tr, contrasts=None, weights=None):
    """Estimate detection power of design matrix.
    
    Parameters
    ----------
    X : array, shape=(n_acq,n_cond)
        Design matrix (i.e. boxcars convolved with HRF).
    tr : float
        Repetition time (in seconds).
    contrasts : array, shape=(n_cond,n_cond)
        Binary matrix denoting pairwise contrasts to estimate. Only 
        reads values from lower diagonal. If None, defaults to all
        pairwise contrasts.
    weights : array, shape=(n_contrasts)
        Weight of each contrast (event and pairwise) in average
        detection power estimate.
        
    Returns
    -------
    R_tot : float
        Detection power.
        
    References
    ----------
    [1] Liu, T. T., & Frank, L. R. (2004). Efficiency, power, and entropy in event-related 
        fmri with multiple trial types: Part i: Theory. NeuroImage, 21(1), 387-400.
    [2] Liu, T. T. (2004). Efficiency, power, and entropy in event-related fMRI with multiple 
        trial types. Part II: design of experiments. NeuroImage, 21(1), 401–413.
    """
    
    ## Compute (inverse) Fisher information matrix.
    finv = np.linalg.inv(X.T @ X)

    ## Compute normalization term.
    hrf = spm_hrf(tr)
    hh = hrf @ hrf

    ## Preallocate space.
    tmp = np.zeros(X.shape[-1])
    R = []
    
    ## Event contrasts.
    for i in range(X.shape[-1]):
        D = _put(tmp, i, 1)
        R = np.append( R, 1. / (np.squeeze(D @ finv @ D.T) * hh) )
        
    ## Pairwise contrasts.
    if contrasts is None: contrasts = np.ones((X.shape[-1], X.shape[-1]))
    contrasts = np.tril(contrasts, k=-1)
    for j, i in np.column_stack(np.where(contrasts)):
        D = _put(tmp, [i,j], [1,-1])
        R = np.append( R, 1. / (np.squeeze(D @ finv @ D.T) * hh) )
        
    ## Compute average.
    R_tot = 1. / np.average(1./R, weights=weights)
        
    return R_tot

def design_efficiency(X, q, k, contrasts=None, weights=None):
    """Estimate design efficiency of design matrix.
    
    Parameters
    ----------
    X : array, shape=(n_acq,n_cond)
        Design matrix (i.e. boxcars convolved with HRF).
    q : int
        Number of conditions.
    k : int
        Size of HRF window.
    contrasts : array, shape=(n_cond,n_cond)
        Binary matrix denoting pairwise contrasts to estimate. Only 
        reads values from lower diagonal. If None, defaults to all
        pairwise contrasts.
    weights : array, shape=(n_contrasts)
        Weight of each contrast (event and pairwise) in average
        detection power estimate.
        
    Returns
    -------
    C_tot : float
        Estimation efficiency.
        
    References
    ----------
    [1] Liu, T. T., & Frank, L. R. (2004). Efficiency, power, and entropy in event-related 
        fmri with multiple trial types: Part i: Theory. NeuroImage, 21(1), 387-400.
    [2] Liu, T. T. (2004). Efficiency, power, and entropy in event-related fMRI with multiple 
        trial types. Part II: design of experiments. NeuroImage, 21(1), 401–413.
    """
    
    assert isinstance(q, int) and isinstance(k, int)
    
    ## Compute (inverse) Fisher information matrix.
    try: finv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError: finv = np.linalg.pinv(X.T @ X)

    ## Preallocate space.
    tmp = np.zeros(q)
    K = np.identity(k)
    C = []
    
    ## Event contrasts.
    for i in range(q):
        L = np.kron(_put(tmp, i, 1), K)
        C = np.append( C, np.trace(L @ finv @ L.T) )
        
    ## Pairwise contrasts.
    if contrasts is None: contrasts = np.tril(np.ones((q,q)), k=-1)
    contrasts = np.tril(contrasts, k=-1)
    for j, i in np.column_stack(np.where(contrasts)):
        L = np.kron(_put(tmp, [i,j], [1,-1]), K)
        C = np.append( C, np.trace(L @ finv @ L.T) )
        
    ## Compute average.
    C_tot = 1. / np.average(C, weights=weights)
        
    return C_tot