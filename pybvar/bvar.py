import numpy as np
from scipy.stats import wishart

def lagdata(data, p, intercept) :
    """ Create lagged data.
    Parameter data: The data to be lagged.
    Parameter p: The number of lags.
    Parameter intercept: A boolean variable indicating whether the model has an intercept or not.
    Returns: A matrix with the lagged data.
    """
    # preliminary stuff
    nl = data.shape[0]
    nv = data.shape[1]
    
    # check for intercept
    if intercept == True:
        constant =1
    else:
        constant = 0
    
    # create new matrix of the appropriate dimension
    ndim = nv * p + constant
    laggeddata = np.empty((nl-p-1,ndim))
    
    if intercept == True:
        laggeddata[:, 0] = 1
    
    for i in range(1, (p+1)):
        ind1 = (i-1)*nv+constant
        ind2 = i*nv+constant
        laggeddata[:,ind1:ind2] = data[(p-i):(nl-1-i),:]
    
    return laggeddata


class bvar:
    """Class for the bayesian VAR model.
    """
    def __init__(self, data, prior):
        """Initializes the VAR model.
            Parameter data: The data used in the model.
            Parameter prior: The prior used for inference.
        """
        # Store model information
        self.data = data
        self.p = prior.p
        self.intercept = prior.intercept
        self.nv = prior.nv
        
        if self.intercept == True:
            constant = 1
        else:
            constant = 0
        
        self.nk = self.nv * self.p + constant
        
        # Store data
        self.y = data[(self.p+1):,:]
        self.x = lagdata(data,self.p,self.intercept)
        
        self.prior = prior
        
    def mcmc(self, nreps, burnin, nthin=1):
        """Estimate a bayesian VAR model using Gibbs-sampling.
        Parameter nreps: total number of draws.
        Parameter burning: number of burn-in draws.
        Parameter nthin: Thinning parameter
        Return: 0 if there were no problems during the run of the MCMC algorithm.
        """
        
        # Declare variables for storage
        self.Betadraws  = np.empty((self.nk, self.nv, int((nreps-burnin)/nthin)))
        self.Sigmadraws = np.empty((self.nv, self.nv, int((nreps-burnin)/nthin)))
        
        # Initialize the Gibbs-Sampler
        draw = self.prior.init_mcmc(self.y,self.x)
        Alpha = draw[0]
        Sigma = draw[1]
        
        # Gibbs sampling
        for ireps in range(1,(nreps + 1)):
            
            # print progress
            if ireps % 100 == 0:
                print(ireps)
            
            # Draw posterior
            draw = self.prior.draw_posterior(self.y,self.x,Alpha,Sigma)
            Alpha = draw[0]
            Sigma = draw[1]
            
            # Store results
            if ireps > burnin and (ireps-burnin) % nthin == 0:
                i = int((ireps - burnin)/nthin - 1) 
                self.Betadraws[:,:,i] = Alpha.reshape((self.nk,self.nv))
                self.Sigmadraws[:,:,i] = Sigma
                
                
        return(0)