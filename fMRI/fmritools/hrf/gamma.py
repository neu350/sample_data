import numpy as np
from scipy.stats import gamma
from scipy.special import gamma as fgamma

def _to_shape_rate(mode, sd):
    '''Convert parameters from gamma(mode, sd) to gamma(shape, rate).'''
    rate = ( mode + np.sqrt( mode**2 + 4*sd**2 ) ) / ( 2 * sd**2 )
    shape = 1 + mode * rate
    return shape, rate

def gamma_mode_pdf(x, mode, sd):
    """Probability density function for (mode, sd)-parameterized gamma.

    Parameters
    ----------
    x : array, shape=(n_obs,)
        Observations.
    mode : float
        Center of distribution.
    sd : float
        Dispersion of distribution.

    Returns
    -------
    pdf : array, shape=(n_obs,)
        Probability density of observation under distribution.
    """

    ## Convert mode/sd to shape/rate.
    shape, rate = _to_shape_rate(mode, sd)

    ## Compute and return PDF.
    return rate ** shape / fgamma(shape) * x ** (shape - 1) * np.exp(-rate * x)

def spm_hrf(TR, t1=6, t2=16, d1=1, d2=1, ratio=6, onset=0, kernel=32):
    """Python implementation of spm_hrf.m from the SPM software.

    Parameters
    ----------
    TR : float
        Repetition time at which to generate the HRF (in seconds).
    t1 : float (default=6)
        Delay of response relative to onset (in seconds).
    t2 : float (default=16)
        Delay of undershoot relative to onset (in seconds).
    d1 : float (default=1)
        Dispersion of response.
    d2 : float (default=1)
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
    [1] Adapted from the Poldrack lab fMRI tools.
        https://github.com/poldracklab/poldracklab-base/blob/master/fmri/spm_hrf.py
    """

    ## Define metadata.
    fMRI_T = 16.0
    TR = float(TR)

    ## Define times.
    dt = TR/fMRI_T
    u  = np.arange(kernel/dt + 1) - onset/dt

    ## Generate (super-sampled) HRF.
    hrf = gamma(t1/d1,scale=1.0/(dt/d1)).pdf(u) - gamma(t2/d2,scale=1.0/(dt/d2)).pdf(u)/ratio

    ## Downsample.
    good_pts=np.array(range(np.int(kernel/TR)))*fMRI_T
    hrf=hrf[good_pts.astype(int)]

    ## Normalize and return.
    hrf = hrf/np.sum(hrf)
    return hrf

def single_gamma_hrf(TR, t=5.4, d=5.2, onset=0, kernel=32):
    """Single gamma hemodynamic response function.

    Parameters
    ----------
    TR : float
        Repetition time at which to generate the HRF (in seconds).
    t : float (default=5.4)
        Delay of response relative to onset (in seconds).
    d : float (default=5.2)
        Dispersion of response.
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
    hrf = (u / t) ** ((t ** 2) / (d ** 2) * 8.0 * np.log(2.0)) \
          * np.e ** ((u - t) / -((d ** 2) / t / 8.0 / np.log(2.0)))

    ## Downsample.
    good_pts=np.array(range(np.int(kernel/TR)))*fMRI_T
    hrf=hrf[good_pts.astype(int)]

    ## Normalize and return.
    hrf = hrf/np.sum(hrf)
    return hrf

def double_gamma_hrf(TR, t1=5.4, t2=10.8, d1=5.2, d2=7.35, ratio=6, onset=0, kernel=32):
    """Double gamma hemodynamic response function.

    Parameters
    ----------
    TR : float
        Repetition time at which to generate the HRF (in seconds).
    t1 : float (default=5.4)
        Delay of response relative to onset (in seconds).
    t2 : float (default=10.8)
        Delay of undershoot relative to onset (in seconds).
    d1 : float (default=5.2)
        Dispersion of response.
    d2 : float (default=7.35)
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

    return single_gamma_hrf(TR,t1,d1,onset,kernel) \
         - single_gamma_hrf(TR,t2,d2,onset,kernel) / ratio
