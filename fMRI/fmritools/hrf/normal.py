import numpy as np
from scipy.stats import norm

def double_gaussian_hrf(TR, t1=5, t2=15, d1=2, d2=4, ratio=6, onset=0, kernel=32):
    """Double normal hemodynamic response function.

    Parameters
    ----------
    TR : float
        Repetition time at which to generate the HRF (in seconds).
    t1 : float (default=5)
        Delay of response relative to onset (in seconds).
    t2 : float (default=15)
        Delay of undershoot relative to onset (in seconds).
    d1 : float (default=2)
        Dispersion of response.
    d2 : float (default=4)
        Dispersion of undershoot.
    ratio : float (default=6)
        Ratio of response to undershoot.
    onset : float (default=0)
        Onset of hemodynamic response (in seconds).
    kernel : float (default=32)
        Length of kernel (in seconds).

    Returns
    -------
    hrf : array
        Hemodynamic repsonse function

    References
    ----------
    [1] Adapted from the pymvpa tools.
        https://github.com/PyMVPA/PyMVPA/blob/master/mvpa2/misc/fx.py
    """

    ## Define metadata.
    fMRI_T = 16.0
    TR = float(TR)

    ## Define times.
    dt = TR/fMRI_T
    u  = np.arange(kernel/dt + 1) - onset/dt
    u *= dt
    
    ## Generate (super-sampled) HRF.
    hrf = norm.pdf(u, t1, d1) - norm.pdf(u, t2, d2) / ratio

    ## Downsample.
    good_pts=np.array(range(np.int(kernel/TR)))*fMRI_T
    hrf=hrf[good_pts.astype(int)]

    ## Normalize and return.
    hrf = hrf/np.sum(hrf)
    return hrf
