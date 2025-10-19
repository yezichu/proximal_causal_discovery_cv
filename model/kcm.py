import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import laplacian_kernel



def kcm(X, res, bsize, alpha,kernel='rbf'):
    """
    The kernel conditional moment (KCM) test. The KCM test is a nonparametric test of the 
    null hypothesis that E[Y-f(Z)|X] = 0. The test is implemented using a Gaussian kernel 
    and a bootstrap procedure to approximate the critical values.

    Args:
        Y (numpy.ndarray): Y is a random variable and may not be ounivariate, eg. dim(Y) = (n, m)
        X (numpy.ndarray): X is a conditional variable and may not be ounivariate, eg. dim(X) = (n, p)
        res (numpy.ndarray): res is a residual vector, eg. Y-\hat{f}(Z). dim(res) = (n, m)
        bsize (int): bootstrap sample size
        alpha (float): significance level

    Returns:
        mh (float): value of the test statistic
        p (float): the p-value
        dec (int): result of hypothesis testing (1: reject, 0: fail to reject)
        bvals (numpy.ndarray): a vector of bootstrap statistics
    """

    # check the compatibility of data dimensions
    X = X.reshape(-1, 1) if X.ndim == 1 else  X
    n = X.shape[0]
    if n != X.shape[0]:
        # exit with an error message
        raise ValueError("KCM: Y and X must have the same number of rows.")

    # calculate the median distance of X and evaluate the kernel k
    
            
    if kernel == 'rbf': 
        K = np.ones((n, n))
        for d in range(X.shape[1]):
            X_d = X[:, d].reshape(-1, 1)
            distX = pairwise_distances(X_d,X_d)
            bw_sq = np.median(distX[np.tril_indices(distX.shape[0],k=-1)]) / 2
            rbf_kernel = RBF(length_scale=bw_sq)
            K_d = rbf_kernel(X_d,X_d)
            K *= K_d
    elif kernel == 'polynomial': 
        K = polynomial_kernel(X,X)
    elif kernel == 'laplace': 
        K = laplacian_kernel(X,X)
    elif kernel == 'binary': 
        K = X.dot(X.T)
        K += (1 - X).dot(1 - X.T)

   
    R = np.einsum('ji,ji->i', res, np.dot(K, res))/ (n * n)
    mh = np.max(R)


    # approximate critical values via bootstrapping
    bvals = np.zeros(bsize)
    for b in range(bsize):
        # draw multinomial random samples
        w = bootstrapper_multinomial(n).reshape(-1, 1)
        # calculate bootstrap test statistic
        Rns = np.einsum('ji,ji->i', w*res, np.dot(K, w*res))
        bvals[b] = np.max(Rns)

    p = np.mean(n * bvals >= n * mh)


    # conduct the test
    dec = 0
    if p < alpha:
        dec = 1  # reject the null hypothesis

    return mh, p, dec, bvals

 

# def bootstrapper_multinomial(n):
#     """
#     Produce a sequence of i.i.d Multinomial(n; 1/n,... 1/n) random variables.
#     This is described on page 5 of Liu et al., 2016 (ICML 2016).
#     """
#     M = np.random.multinomial(n, np.ones(n) / n)
#     return M/n - 1/n


def bootstrapper_multinomial(n):
    """
    Produce a sequence of i.i.d Multinomial(n; 1/n,... 1/n) random variables.
    This is described on page 5 of Liu et al., 2016 (ICML 2016).
    """
    M = 2.0*np.random.randint(0, 1+1, n)-1.0
    return M/n







