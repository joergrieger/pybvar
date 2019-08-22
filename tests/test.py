import numpy as np
from pybvar.bvar import bvar
from pybvar.priors import Uninformative
from pybvar.priors import ConjugateNormal

# Create Testdata
TestData = np.random.standard_normal((150, 4))

# Test uninformative prior
prior1 = Uninformative(TestData, 2, False)
bv = bvar(TestData, prior1)
bv.mcmc(1000, 200, 1)

# Test conjugate-normal prior
prior2 = ConjugateNormal(TestData, 2, False, 1, 0.1, 0.1, 10)
bv2 = bvar(TestData, prior2)
bv2.mcmc(1000, 200, 1)
