#import theano.tensor as tt
#from pymc3 import DensityDist, Uniform, Model
#
#def star(x):
##    return -0.5 * tt.exp(-tt.sum(x ** 2))
#    # or if you need the components individually
#    return -0.5 * tt.exp(-x[0] ** 2 - x[1] ** 2)
#
#with Model() as model:
#    lim = 3
#    x0 = Uniform('x0', -lim, lim)
#    x1 = Uniform('x1', -lim, lim)
#
#    x = tt.stack([x0,x1])
#    # Create custom densities
#    star = DensityDist('star', star)
#    
#N = 100
#samples = star.random(size=100)








## Example from 
#import arviz as az
#import pymc3 as pm 
#import numpy as np
#import matplotlib.pyplot as plt
#
#with pm.Model() as model:
#    idx = pm.Uniform('idx', 0, 1)
#    a = pm.Uniform('a', np.array([-15, 0]), np.array([-5, 5]), shape=2)
#    b = pm.Deterministic('b', pm.math.switch(idx < 0.25, a[0], a[1]))
#    step = pm.Metropolis()
#    trace = pm.sample(1000, step)
# 
#ax = az.plot_trace(trace, var_names="a")
#ax[0, 0].axvline(0.5, label="True value", color="k")
#ax[0, 0].legend()
#plt.suptitle("Sampling distribution")
#plt.show()



## Making this example working: https://stackoverflow.com/questions/43712106/sampling-multivariate-uniform-in-pymc3
## https://discourse.pymc.io/t/can-pymc3-get-samples-from-a-not-posterior-complex-distribution-using-mcmc/3499/14
#import arviz as az
#import pymc3 as pm 
#import numpy as np
#import matplotlib.pyplot as plt
#import theano.tensor as tt
#import seaborn as sbn
#
#mu, sigma = 10., 2.5
#input_size = 2
#input_test = [0] * input_size
#
#def star():
#    def star_ahah(x):
#        return -tt.sum(0.5 * (tt.log(2 * np.pi * sigma ** 2) + ((x - mu) / sigma) ** 2))
#    return star_ahah
#
#
#    
#with pm.Model() as model:
#    lim = 30
#    x0 = pm.Uniform('x0', -lim, lim)
#    x1 = pm.Uniform('x1', -lim, lim)
#    x = np.array([x0,x1])
#    b = pm.DensityDist('star', star(), shape=input_size, testval=input_test)
#    step = pm.Metropolis()
#    trace = pm.sample(1000, step=step, cores=1)
#
#
#
##ax = az.plot_trace(trace, var_names=["x0", "x1"])
##ax[0, 0].axvline(0.5, label="True value", color="k")
##ax[0, 0].legend()
##plt.suptitle("Sampling distribution")
##plt.show()
#    
#sbn.kdeplot(trace['x0'][500::100])
#sbn.kdeplot(trace['x1'][500::100])
#plt.show()






# Log Likelyhood: https://discourse.pymc.io/t/custom-black-box-likelihood-example-not-working/5544
import arviz as az
import pymc3 as pm 
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import seaborn as sbn

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
theano.config.exception_verbosity='high'


alpha, sigma = 5, 1
size = 10000
# Simulate outcome variable
Y = alpha + np.random.randn(size)*sigma

def my_loglike(theta):
    alpha,sigma = theta
    return -(0.5/sigma**2)*np.sum((alpha - Y)**2)

class LogLike(tt.Op):
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables
        # call the log-likelihood function
        logl = self.likelihood(theta)
        outputs[0][0] = np.array(logl) # output the log-likelihood

loglik = LogLike(my_loglike)

with pm.Model() as model:
    alpha = pm.Uniform('alpha', 0, 25)
    sigma = pm.Uniform('sigma', 0.1, 2)

    # Create likelihood
    theta = tt.as_tensor_variable([alpha, sigma])
    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist('likelihood', lambda v: loglik(v), observed={'v': theta})
    trace = pm.sample(1000)
