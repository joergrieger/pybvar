# pybvar

'pybvar' is a package for bayesian vector autoregression in Python. This package is similar to [bvars](https://github.com/joergrieger/bvars).

This readme contains some examples on the usage of the package.

The package is in a very preliminary stage of its development.

# Examples

Let's say that we would like to estimate a bayesian VAR with an uninformative prior.

The following code sets up the uninformative prior for a VAR model with lag p=2 and an intercept

```
prior = uninformative(data,2,True)
```

In the next step we have to create a bvar object and pass the prior to it. This is done using the following code:

```
bv = bvar(data,prior)
```

To start the mcmc algorithm with 10,000 draws and 5,000 burn-in draws and we only want to keep every 5th draw we have to use the following code

```
results = bv.mcmc(10000,5000,5)
```