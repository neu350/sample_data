import numpy as np
from ..hrf import spm_hrf

def _convolve(a, v, mode='full'):
    return np.convolve(a, v, mode)

def design_matrix(tr, n_acq, events, hrf=None, normalize=False,
                  period=0.001, return_boxcars=False):
    """Generate fMRI design matrix.

    Parameters
    ----------
    tr : float
        Repetition time.
    n_acq : int
        Number of acquisitions.
    events : array, shape=(n_events,3)
        Neural events. The first column contains the event onset time,
        the second column contains the event offset time, and the third
        column contains the event condition.
    hrf : function (default = spm_hrf)
        Function generating the HRF (must accept TR).
    normalize : bool
        If true, normalize regressors to amplitude = 1.
    period : float
        Super-sampling resolution (in seconds).
    return_boxcars : bool
        If true, return boxcars.

    Returns
    -------
    t : array, shapes=(n_acq,)
        TR onsets (in seconds).
    X : array, shape=(n_acq,n_cond)
        Design matrix (i.e. boxcars convolved with HRF).
    Z : array, shape=(n_acq,n_cond) (optional)
        Design matrix boxcars.
    """

    ## Error-catching.
    assert events.shape[-1] == 3
    events = np.copy(events)
    _, events[:,2] = np.unique(events[:,2], return_inverse=True)

    ## Define (super-sampled) times.
    sst = np.arange(0, tr * n_acq, period)

    ## Define boxcars.
    boxcars = np.zeros((sst.size, int(events[:,2].max()) + 1))
    for onset, offset, cond in events:
        boxcars[np.logical_and(sst >= onset, sst < offset),int(cond)] = 1

    ## Define HRF.
    if hrf is None: hrf = spm_hrf(period)
    else: hrf = hrf(period)

    ## Perform convolution.
    X = np.apply_along_axis(_convolve, 0, boxcars, hrf)[:sst.size]

    ## Downsampling.
    t = np.arange(0, tr * n_acq, tr)
    boxcars = boxcars[np.in1d(sst, t)]
    X = X[np.in1d(sst, t)]

    ## Normalize.
    if normalize: X /= X.max(axis=0)

    if return_boxcars: return t, X, boxcars
    else: return t, X

def fir_matrix(tr, n_acq, events, kernel=None):
    """Generate FIR design matrix.

    Parameters
    ----------
    tr : float
        Repetition time.
    n_acq : int
        Number of acquisitions.
    events : array, shape=(n_events,2)
        Neural events. The first column contains the event onset index,
        and the second column contains the event condition.
    kernel : int
        Length of HRF (in acquisitions). If None, defaults to the
        approximate number of TRs that occur in 16s.

    Returns
    -------
    t : array, shapes=(n_acq,)
        TR onsets (in seconds).
    X : array, shape=(n_acq,n_cond*kernel)
        Design matrix (i.e. boxcars convolved with HRF).
    """

    ## Error-catching.
    assert events.shape[-1] == 2
    events = np.copy(events).astype(int)
    _, events[:,1] = np.unique(events[:,1], return_inverse=True)

    ## Define kernel width.
    if kernel is None: k = int(16. / tr)
    else: k = int(kernel)

    ## Define times.
    t = np.arange(0, tr * n_acq, tr)

    ## Preallocate space.
    q = events[:,1].max() + 1
    X = np.zeros((t.size, k*q))

    ## Iteratively add events.
    for onset, cond in events:
        i = np.arange(onset,onset+k)        # Row indices
        j = np.arange(cond*k,(cond+1)*k)    # Col indices
        X[i[i<n_acq],j[i<n_acq]] = 1

    return t, X
