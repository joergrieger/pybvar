import numpy as np
from pybvar.bvar import bvar
from pybvar.priors import uninformative

# Create Testdata
TestData = np.random.standard_normal((100, 4))
prior = uninformative(TestData, 2, False)
bv = bvar(TestData, prior)


def test_mcmc():
    assert bv.mcmc(20, 10, 1) == 0