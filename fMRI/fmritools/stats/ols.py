import numpy as np
from scipy.stats import t as tdist
from scipy.stats import f as fdist

class OLS(object):
    """Fit ordinary least squares (OLS) model to data.

    Parameters
    ----------
    Y : array, shape (n_times, n_voxels)
        fMRI observations.
    X : array, shape (n_times, n_regressors)
        Design matrix.
    Z : array, shape (n_times, m_regressors)
        Nuisance regressors.
    intercept : True | False
        If true, include intercept term in nuisance regressors.

    Attributes
    ----------
    n_times : int
        Number of acquisitions.
    n_voxels : int
        Number of voxels.
    n_coef : int
        Number of parameters.
    n_nuis : int
        Number of nuisance parameters.
    coef : array, shape (n_regressors, n_voxels)
        Parameter estimates corresponding to the design matrix.
    stderr : array, shape (n_regressors, n_voxels)
        The standard errors of the parameter estimates.
    ssr : array, shape (n_voxels)
        Sum of squared residuals.
    tvalues : array, shape (n_regressors, n_voxels)
        The t-statistics of the parameter estimates.
    pvalues : array, shape (n_regressors, n_voxels)
        The p-values of the parameter estimates.
    nuisance: array, shape (m_regressors, n_voxels)
        Regression coefficients corresponding to nuisance regressors.
    """

    def __init__(self, Y, X, Z=None, intercept=False):

        ## Prepare observations.
        if np.ndim(Y) == 1: Y = np.reshape(Y, (-1,1))
        elif np.ndim(Y) == 2: Y = np.asarray(Y)
        else: raise ValueError('Y must be 2d at most.')

        ## Prepare design matrix.
        if np.ndim(X) == 1: X = np.reshape(X, (-1,1))
        elif np.ndim(X) == 2: X = np.asarray(X)
        else: raise ValueError('X must be 2d at most.')

        ## Prepare nuisance regressors.
        if Z is None: Z = np.empty((X.shape[0], 0))
        elif np.ndim(Z) == 1: Z = np.reshape(Z, (-1,1))
        elif np.ndim(Z) == 2: Z = np.asarray(Z)
        else: raise ValueError('Z must be 2d at most.')

        ## Include intercept.
        if intercept: Z = np.column_stack([np.ones(np.shape(Z)[0]), Z])

        ## Error-catching.
        assert np.all(np.diff((X.shape[0], Y.shape[0], Z.shape[0])) == 0)
        self._Y = Y; self._X = X; self._Z = Z

        ## Store metadata.
        self.n_times, self.n_voxels = np.shape(Y)
        _, self.n_coef = np.shape(X)
        _, self.n_nuis = np.shape(Z)
        self._fit = False

    def __repr__(self):
        return '<Ordinary least squares>'

    def fit(self, pvalues=True):
        """Fit OLS model to data.

        Parameters
        ----------
        pvalues : True | False
            If true, compute p-values corresponding to t-statistics.
        """

        ## Assemble data.
        X = np.column_stack([self._X, self._Z])
        Y = self._Y

        ## Perform linear regression.
        beta, ssr, _, _ = np.linalg.lstsq(X, Y, rcond=-1)
        self.coef = beta[:self.n_coef]; self.nuisance = beta[self.n_coef:]
        self.ssr = ssr

        ## Compute scale.
        self._scale = self.ssr / (self.n_times - (self.n_coef + self.n_nuis))

        ## Compute standard error.
        cov = np.diag(np.linalg.inv(X.T @ X))
        bse = np.sqrt(np.outer(cov, self._scale))
        self.stderr = bse[:self.n_coef]

        ## Compute t-statistics.
        self.tvalues = self.coef / self.stderr

        ## Compute p-values.
        if pvalues: self.pvalues = tdist.sf(np.abs(self.tvalues), self.n_times-1)*2

        self._fit = True
        return self

    def f_test(self, contrast, pvalues=True):
        """Compute contrast statistics.

        Parameters
        ----------
        contrast : array, shape (n_coef) or (n_coef, n_coef)
            The regression contrast matrix. See notes for details.
        pvalues : True | False
            If true, compute p-values corresponding to F-statistics.

        Returns
        -------
        F : array, shape (n_voxels)
            F-statistics for corresponding contrast.
        pvalues : array, shape (n_voxels)
            P-values corresponding to F-statistics.

        Notes
        -----
        The contrast vector or matrix determines the type of contrast that is performed.
        When a vector is supplied, a t-contrast is performed. For example, contrast = [1, -1]
        computes a test of differences bewteen the first and second regression coefficients.
        Note that the contrast vector is not normalized; if the contrast vector needs to sum
        to zero, that needs to be done outside of the function.

        When a matrix is supplied, a F-contrast is performed. For example, passing the
        identity matrix tests whether any of the coefficients is significantly different
        from zero.
        """

        if not self._fit:
            raise ValueError('OLS model must be fit first.')

        ## Prepare contrast.
        if np.ndim(contrast) == 1:
            C = np.expand_dims(contrast, 0)
        elif np.ndim(contrast) == 2:
            C = np.asarray(contrast)
        else:
            raise ValueError('Contrast must be 1d or 2d.')

        ## Error-catching.
        np.testing.assert_array_equal(C.shape[-1], self.n_coef)

        ## Precompute coefficient contrast.
        Cw = C @ self.coef

        ## Precompute coefficient covariance.
        finv = np.linalg.inv(self._X.T @ self._X)
        Sigma = np.einsum('i,jk->ijk', self._scale, finv)

        ## Precompute contrast rank.
        rank = np.linalg.matrix_rank(C)

        ## Iteratively compute F-statistics.
        ## TODO: Improve efficiency.
        F = np.zeros(self.n_voxels)
        for i, (cw, cov) in enumerate(zip(Cw.T, Sigma)):
            F[i] = cw.T @ np.linalg.pinv(rank * C @ cov @ C.T) @ cw

        ## Compute p-values.
        if pvalues:

            ## Define degrees of freedom.
            df_num = rank
            df_denom = self.n_times - self.n_coef

            ## Compute p-values.
            pvalues = fdist.sf(F, df_num, df_denom)

            return F, pvalues

        else:

            return F
