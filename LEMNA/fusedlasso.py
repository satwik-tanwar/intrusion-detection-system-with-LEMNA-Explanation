from sklearn.linear_model import Lasso
import numpy as np
from scipy.sparse import diags

def fused_lasso(X, y, alpha1, alpha2):
    n_features = X.shape[1]

    # create fused lasso model and fit to data
    model = Lasso(alpha=alpha1, fit_intercept=False)
    model.fit(X, y)

    # get Lasso coefficients
    coef = model.coef_

    # apply fused lasso penalty
    D = diags([1, -1], [0, 1], shape=(n_features-1, n_features)).toarray()
    D = np.vstack([D, np.zeros((1, n_features))])
    D[0, 0] = 1
    fused_coef = np.linalg.inv(X.T @ X + (alpha2 / alpha1) * D.T @ D) @ X.T @ y

    return fused_coef
