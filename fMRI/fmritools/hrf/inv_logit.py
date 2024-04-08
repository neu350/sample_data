import numpy as np

def inv_logit(arr):
    return 1 / (1 + np.exp(-arr))

def inv_logit_hrf(TR, t1=3, t2=8, t3=16, d1=0.5, d2=2, d3=2, ratio=6, onset=0, kernel=32):
    """Inverse logistic hemodynamic response function.

    Parameters
    ----------
    TR : float
        Repetition time at which to generate the HRF (in seconds).
    t1 : float (default=3)
        Midpoint of response relative to onset (in seconds).
    t2 : float (default=8)
        Midpoint of undershoot relative to onset (in seconds).
    t3 : float (default=16)
        Midpoint of recovery relative to onset (in seconds).
    d1 : float (default=0.5)
        Dispersion of response.
    d2 : float (default=2)
        Dispersion of undershoot.
    d3 : float (default=2)
        Dispersion of recovery.
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
    [1] Lindquist, M. A., & Wager, T. D. (2007). Validity and power in hemodynamic response
        modeling: a comparison study and a new approach. Human brain mapping, 28(8), 764-784.
    """

    ## Define metadata.
    fMRI_T = 16.0
    TR = float(TR)

    ## Define times.
    dt = TR/fMRI_T
    u  = np.arange(kernel/dt + 1) - onset/dt
    u *= dt
    
    ## Define amplitudes.
    a1 = 1
    a2 = a1 + a1 / ratio
    a3 = a2 - a1

    ## Define HRF.
    hrf = inv_logit((u-t1)/d1) - a2 * inv_logit((u-t2)/d2) + a3 * inv_logit((u-t3)/d3)

    ## Downsample.
    good_pts=np.array(range(np.int(kernel/TR)))*fMRI_T
    hrf=hrf[good_pts.astype(int)]

    ## Normalize and return.
    hrf = hrf/np.sum(hrf)
    return hrf
