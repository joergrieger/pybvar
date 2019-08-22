import numpy as np
from scipy.stats import wishart

class Uninformative:
    """Class for the uninformative prior."""
    def __init__(self, data, p, intercept):
        """Initialize the uninformative prior.
        Parameter data: A matrix of dimension Txk containing the data.
        Parameter p: The number of lags in the model.
        Parameter intercept: Boolean variable whether the model contains an intercept or not
        """
        # Model information
        self.intercept = intercept
        self.T = data.shape[0]
        self.p = p
        self.nv = data.shape[1]

        
    def draw_posterior(self, y, x, alpha, sigma):
        """One draw from the posterior for the uninformative prior.
        Parameter y: A (T-p)xk matrix with the LHS of the model. With T being the length of the original data set, p the number of lags and k the number of variables in the model.
        Parameter x: A (T-p)x(k*p+constant) matrix with the RHS of the model. With T being the length of the original data, p the number of lags, k the number of variables in the model and constant is either 0 or 1 depending whether the model has an intercept (constant = 1) or not (constant = 0).
        Parameter alpha: The regression coefficients of the previous draw.
        Parameter sigma: The variance-covariance matrix from the previous draw.
        Returns: One draw of the mcmc consists of the draw for the coefficients and a draw for the variance-covariance matrix.
        """
        # get OLS estimates
        tmp1 = np.linalg.inv(np.matmul(np.transpose(x), x))
        tmp2 = np.matmul(np.transpose(x), y)
        olsestim = np.matmul(tmp1, tmp2)
        
        # residuals 
        resids = y - np.matmul(x, olsestim)
        sse = np.matmul(np.transpose(resids), resids)
        
        # Draw coefficients (beta)
        vpost = np.kron(sigma, np.linalg.inv(np.matmul(np.transpose(x), x)))
        alpha = np.random.multivariate_normal(np.ndarray.flatten(olsestim), vpost)
        
        # Draw variance-covariance matrix
        sigma = wishart.rvs(self.T, np.linalg.inv(sse))
        
        # Return Values
        return alpha, sigma
    
    def init_mcmc(self, y, x):
        """ Initializes the mcmc algorithm for the uninformative prior.
        Parameter y: A (T-p)xk matrix with the LHS of the model. With T being the length of the original data set, p the number of lags and k the number of variables in the model.
        Parameter x: A (T-p)x(k*p+constant) matrix with the RHS of the model. With T being the length of the original data, p the number of lags, k the number of variables in the model and constant is either 0 or 1 depending whether the model has an intercept (constant = 1) or not (constant = 0).
        Returns: The initial draw of the mcmc algorithm for a model with uninformative prior.
        """
        # get OLS estimates
        tmp1 = np.linalg.inv(np.matmul(np.transpose(x), x))
        tmp2 = np.matmul(np.transpose(x), y)
        olsestim = np.matmul(tmp1, tmp2)
        
        # residuals
        resids = y - np.matmul(x,olsestim)
        sse = np.matmul(np.transpose(resids), resids)
        
        # Draw Sigma
        sigma = wishart.rvs(self.T, np.linalg.inv(sse))
        alpha = None
        retlist = (alpha, sigma)
        return retlist

class ConjugateNormal:
    """
    Class for the Conjugate-Normal prior.
    Parameter data: A matrix of dimension Txk containing the data.
    Parameter p: The number of lags in the model.
    Parameter intercept: Boolean variable whether the model contains an intercept or not
    Parameter coefprior prior on the VAR-coefficients (scalar)
    Parameter coefpriorvar prior on the variance of the VAR-coefficients (scalar)
    Parameter varprior prior on the variance (scalar)
    Parameter varpriordof degree of freedom of the variance
    """
    def __init__(self, data, p, intercept, coefprior, coefpriorvar, varprior, varpriordof):
        self.intercept = intercept
        self.p = p
        self.T = data.shape[0]
        self.nv = data.shape[1]

        if intercept:
            constant = 1
        else:
            constant = 0

        self.coefprior = coefprior * np.ones((self.nv * self.p + constant, self.nv))
        self.coefpriorvar = coefpriorvar * np.eye(self.nv * self.p + constant)

        self.varprior = np.eye(self.nv)
        self.varpriordof = varpriordof

    def init_mcmc(self, y, x):
        """ Initializes the mcmc algorithm for the Conjugate-Normal prior.
        Parameter y: A (T-p)xk matrix with the LHS of the model. With T being the length of the original data set, p the number of lags and k the number of variables in the model.
        Parameter x: A (T-p)x(k*p+constant) matrix with the RHS of the model. With T being the length of the original data, p the number of lags, k the number of variables in the model and constant is either 0 or 1 depending whether the model has an intercept (constant = 1) or not (constant = 0).
        Returns: The initial draw of the mcmc algorithm for a model with uninformative prior.
        """
        # get OLS estimates
        tmp1 = np.linalg.inv(np.matmul(np.transpose(x), x))
        tmp2 = np.matmul(np.transpose(x), y)
        olsestim = np.matmul(tmp1, tmp2)

        # residuals
        resids = y - np.matmul(x, olsestim)
        sse = np.matmul(np.transpose(resids), resids)

        # Draw Sigma
        sigma = wishart.rvs(self.T, np.linalg.inv(sse))
        alpha = np.ndarray.flatten(olsestim)
        retlist = (alpha, sigma)
        return retlist

    def draw_posterior(self, y, x, alpha, sigma):
        """One draw from the posterior for the Conjugate-Normal prior.
        Parameter y: A (T-p)xk matrix with the LHS of the model. With T being the length of the original data set, p the number of lags and k the number of variables in the model.
        Parameter x: A (T-p)x(k*p+constant) matrix with the RHS of the model. With T being the length of the original data, p the number of lags, k the number of variables in the model and constant is either 0 or 1 depending whether the model has an intercept (constant = 1) or not (constant = 0).
        Parameter alpha: The regression coefficients of the previous draw.
        Parameter sigma: The variance-covariance matrix from the previous draw.
        Returns: One draw of the mcmc consists of the draw for the coefficients and a draw for the variance-covariance matrix.
        """
        # OLS estimates
        tmp1 = np.linalg.inv(np.matmul(np.transpose(x), x))
        tmp2 = np.matmul(np.transpose(x), y)
        olsestim = np.matmul(tmp1, tmp2)
        resids = y - np.matmul(x, olsestim)
        sse = np.matmul(np.transpose(resids), resids)

        # Posterior for coefficients
        vpost = np.linalg.inv(np.linalg.inv(self.coefpriorvar) + np.matmul(np.transpose(x), x))
        apost = np.matmul(vpost, np.matmul(np.linalg.inv(self.coefpriorvar), self.coefprior) + np.matmul(np.matmul(np.transpose(x), x), olsestim))
        cova = np.kron(sigma, vpost)
        alpha = np.random.multivariate_normal(np.ndarray.flatten(apost), cova)

        # Posterior for variance - covariance matrix
        vpost = self.T + self.varpriordof
        tmp1 = sse + self.varprior + np.matmul(np.transpose(olsestim), np.matmul(np.transpose(x), np.matmul(x, olsestim)))
        tmp2 = np.matmul(np.matmul(np.transpose(self.coefprior), np.linalg.inv(self.coefpriorvar)), self.coefprior)
        tmp3 = np.matmul(np.matmul(np.transpose(apost), np.linalg.inv(self.coefpriorvar) + np.matmul(np.transpose(x), x)), apost)
        spost = tmp1 + tmp2 + tmp3
        sigma = wishart.rvs(vpost,np.linalg.inv(spost))

        # Return estimates
        retlist = (alpha,sigma)
        return(retlist)



class Minnesota:
    def __init__(self, data, p, intercept):
        
