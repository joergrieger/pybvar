import numpy as np
from scipy.stats import wishart

class uninformative:
    """Class for the uninformative prior."""
    def __init__(self,data,p,intercept):
        """Initialize the uninformative prior.
        Parameter data: A matrix of dimension Txk containing the data.
        Parameter p: The number of lags in the model.
        Parameter intercept: Boolean variable whether the model contains an intercept or not
        """
        # Model information
        self.intercept = intercept
        self.T  = data.shape[0]
        self.nv = data.shape[1]
        self.p  = p
        
    def draw_posterior(self,y,x,Alpha,Sigma):
        """One draw from the posterior for the uninformative prioer.
        Parameter y: A (T-p)xk matrix with the LHS of the model. With T being the length of the original data set, p the number of lags and k the number of variables in the model.
        Parameter x: A (T-p)x(k*p+constant) matrix with the RHS of the model. With T being the length of the original data, p the number of lags, k the number of variables in the model and constant is either 0 or 1 depending whether the model has an intercept (constant = 1) or not (constant = 0).
        Parameter Alpha: The regression coefficients of the previous draw.
        Parameter Sigma: The variance-covariance matrix from the previous draw.
        Returns: One draw of the mcmc consists of the draw for the coefficients and a draw for the variance-covariance matrix.
        """
        # get OLS estimates
        tmp1 = np.linalg.inv(np.matmul(np.transpose(x), x))
        tmp2 = np.matmul(np.transpose(x),y)
        olsestim = np.matmul(tmp1,tmp2)
        
        # residuals 
        resids = y - np.matmul(x,olsestim)
        sse    = np.matmul(np.transpose(resids),resids)
        
        # Draw coefficients (beta)
        Vpost = np.kron(Sigma,np.linalg.inv(np.matmul(np.transpose(x),x)))
        Alpha = np.random.multivariate_normal(np.matrix.flatten(olsestim),Vpost)       
        
        # Draw variance-covariance matrix
        Sigma = wishart.rvs(self.T,np.linalg.inv(sse))
        
        # Return Values
        retlist=(Alpha,Sigma)
        return(retlist)
    
    def init_mcmc(self,y,x):
        """ Initializes the mcmc algorithm for the uninformative prior.
        Parameter y: A (T-p)xk matrix with the LHS of the model. With T being the length of the original data set, p the number of lags and k the number of variables in the model.
        Parameter x: A (T-p)x(k*p+constant) matrix with the RHS of the model. With T being the length of the original data, p the number of lags, k the number of variables in the model and constant is either 0 or 1 depending whether the model has an intercept (constant = 1) or not (constant = 0).
        Returns: The initial draw of the mcmc algorithm for a model with uninformative prior.
        """
        # get OLS estimates
        tmp1 = np.linalg.inv(np.matmul(np.transpose(x), x))
        tmp2 = np.matmul(np.transpose(x),y)
        olsestim = np.matmul(tmp1,tmp2)
        
        # residuals
        resids = y - np.matmul(x,olsestim)
        sse    = np.matmul(np.transpose(resids),resids)
        
        # Draw Sigma
        Sigma = wishart.rvs(self.T,np.linalg.inv(sse))
        Alpha = None
        retlist=(Alpha,Sigma)
        return(retlist)