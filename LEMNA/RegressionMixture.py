import os
import argparse

import numpy as np
from scipy.stats import multivariate_normal
from scipy import io
from LEMNA import fusedlasso

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky


class GMM(GaussianMixture):
    def __init__(
        self,
        y,
        n_components=1,
        *,
        covariance_type='full',
        tol=0.001,
        reg_covar=1e-06,
        max_iter=100,
        n_init=1,
        init_params='kmeans',
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10
    ):

        super(GMM, self).__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol, reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval
        )
        self.Y = np.array(y)
        self.y = np.array(y)

    """Customized m-step to fit fused lasso"""

    def _m_step(self, X, log_resp):
        """ M step.
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
            log_resp : array-like, shape (n_samples, n_components)
                Logarithm of the posterior probabilities (or responsibilities) of
                the point of each sample in X.
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        self.weights_, self.mu, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )

        # update lasso coefficient
        r_ic = np.exp(log_resp)

        for i in range(self.n_components):
            idx = np.where(np.argmax(r_ic, axis=1) == i)

            # ensure it can be fitted by fused lasso
            if len(idx[0]) > (n_samples/(2*self.n_components)):
                data_X_i = X[idx[0]]
                data_Y_i = self.Y[idx[0]]
                n, p = data_X_i.shape

                result = fusedlasso.fused_lasso(
                    data_X_i, data_Y_i, alpha1=1, alpha2=0.5)

                mu_i = np.multiply(result, np.mean(data_X_i, axis=0))
                if i == 0:
                    self.means_ = mu_i
                else:
                    self.means_ = np.vstack((self.means_, mu_i))

            else:
                if i == 0:
                    self.means_ = self.mu[i]
                else:
                    self.means_ = np.vstack((self.means_, self.mu[i]))

        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)
