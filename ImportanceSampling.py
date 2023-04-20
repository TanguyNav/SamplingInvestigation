# -*- coding: utf-8 -*-
""" A script for investigating importance sampling in high dimensional space"""

__authors__ = "tnavez"
__contact__ = "tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 05 2022"

# Libraries
import numpy as np
import scipy.stats as stats
from scipy.integrate import nquad 
from sklearn.neighbors import KDTree, KernelDensity
import matplotlib.pyplot as plt
import random

# Local libraries
from SamplingBenchmark import sample

BOUNDS = [[-30, 30] for i in range(2)]
N_SAMPLES = 5000
SAMPLING_METHOD = "ScrambledHalton"
FUNCTION_NAME = "ShiftedAckley"



# Main function
def main(args=None):    
            
    """ Without importance sampling"""
    print(">>> Function evaluation with many samples starts ...")
    
    # Sample an uniformly distributed sequence
    uniform_sampling = sample(BOUNDS, method = SAMPLING_METHOD, n_samples = N_SAMPLES)
    
    # Compute f(x)
    f_x = []
    for x in uniform_sampling:
        f_x.append(f(x, function_name = FUNCTION_NAME))
        
    # Plot 3D results with uniform sampling
    if len(BOUNDS) == 2:
        plot_results(uniform_sampling, f_x, function_name = FUNCTION_NAME, sampling_method = SAMPLING_METHOD, x_axis=0, y_axis=1)
    
    print(">>> ... evaluation done ")
                  
    """ Smart importance sampling"""
    print("Smart sampling starts.")
    
    ### Init: Build the initial random sampling to get an idea of the profile of the f function. 
    # Sample a reduced uniformly distributed sequence
    reduced_uniform_sampling = sample(BOUNDS, method = SAMPLING_METHOD, n_samples = N_SAMPLES // 50)
    
    # Compute black box f(x) as the target distribution
    estimated_f_x = []
    for x in reduced_uniform_sampling:
        estimated_f_x.append(f(x, function_name = FUNCTION_NAME))
        
    # Plot 3D results with uniform sampling
    if len(BOUNDS) == 2:
        plot_results(reduced_uniform_sampling, estimated_f_x, function_name = FUNCTION_NAME, sampling_method = SAMPLING_METHOD, x_axis=0, y_axis=1)
        
        # Plot distribution histogram
        plot_3D_distribution(reduced_uniform_sampling, BOUNDS)
        
    ### Resample using knowledge of f 
    # Step 1: Weight each point according to f value
#    weights = [1/(N_SAMPLES // 10) for k in range(N_SAMPLES // 10)] # Probability of each sample    
    weights = weight_according_to_neighbours(reduced_uniform_sampling, estimated_f_x, k = N_SAMPLES // 50 // 50)
    weighted_f_x = [weights[k] for k in range(len(estimated_f_x))]
    
    # Plot weighted density
    if len(BOUNDS) == 2:
        fig = plt.figure()
        axs = []
        x = [reduced_uniform_sampling[k][0] for k in range(len(reduced_uniform_sampling))]
        y = [reduced_uniform_sampling[k][1] for k in range(len(reduced_uniform_sampling))]
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(x, y, weighted_f_x, marker="x", s = 0.5)
        ax.set_title("Weighted f_x")
        axs.append(ax)
        plt.show()
    
    # Step 2: Interpolate function for getting a continuous weighted probability density function  
    f_rbf = interpolate_from_samples(reduced_uniform_sampling, weighted_f_x, method = "RBF")
     
    # Plot interpolation results
    fig = plt.figure()
    axs = []
    test_samples = sample(BOUNDS, method = SAMPLING_METHOD, n_samples = N_SAMPLES)
    a_test = []
    for dim in range(len(test_samples[0])):
        a_test.append([test_samples[k][dim] for k in range(len(test_samples))])
    f_test = f_rbf(*a_test)        
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(a_test[0], a_test[1], f_test, marker="x", s = 0.5)
    ax.set_title("RBF interpolation - multiquadrics")
    axs.append(ax)
    plt.show()
     
    
    # Step3-V1: CDF estimation using empirical distribution function
#    # Integrate interpolated function 
#    def f_RBF(*X):
#        return f_rbf(*X)
#    total_area = integrate(f_RBF, BOUNDS)
#    print("Total area under curve is:", total_area)
    
    # TODO: use integration as a metric for computing CDF: This is computationally way to expensive
    
    
#    # Step3-V2: CDF estimation using empirical distribution function
#    P, values = ecdf(BOUNDS, f_rbf, N_SAMPLES // 50) 
##    print("Joint cumulative distribution function :", P)
#        
#    # Plot eCDF
#    fig = plt.figure()
#    axs = []
#    x = []
#    y = []
#    p_values = []
#    for index, p in np.ndenumerate(P):
#        x.append(values[0][index[0]])
#        y.append(values[1][index[1]])
#        p_values.append(p)
# 
#    ax = fig.add_subplot(1, 1, 1, projection='3d')
#    ax.scatter(x, y, p_values, marker="x", s = 0.5)
#    ax.set_title("Empirical CDF")
#    axs.append(ax)
#    plt.show()
#    
#    # Building a Tree structure for quick inverse map queries
#    # Build dictionnary of coordinates by probability value
#    proba_dict = {}
#    p_values = []
#    for index, p in np.ndenumerate(P):
#        if p in p_values:
#            proba_dict[p].append([values[i][index[i]] for i in range(len(index))])
#        else:
#            proba_dict[p] = [[values[i][index[i]] for i in range(len(index))]]
#            p_values.append(p)
#            
#    
#    # Sort by probability
#    p_values = sorted(p_values)
#    sorted_p = {}
#    for i in range(len(p_values)):
#        sorted_p[p_values[i]] = proba_dict[p_values[i]]
#        p_values[i] = [p_values[i]]
#    print("sorted_p:", sorted_p)
#        
#        
#    # Build KDTree on probabilities     
#    kdt = KDTree(p_values, leaf_size=30, metric = "manhattan")
#    
#    # Step5: Uniform sampling of sequence between 0 and 1 followed by inverse map
#    importance_samples = inverse_map_cdf(N_SAMPLES, BOUNDS, sorted_p, kdt)
#    
#    # Compute black box f(x) as the target distribution
#    estimated_f_x = []
#    for x in importance_samples:
#        estimated_f_x.append(f(x, function_name = FUNCTION_NAME))
#        
#    # Plot 3D results with uniform sampling
#    if len(BOUNDS) == 2:
#        plot_results(importance_samples, estimated_f_x, function_name = FUNCTION_NAME, sampling_method = SAMPLING_METHOD, x_axis=0, y_axis=1)
#        
#        # Plot distribution histogram
#        plot_3D_distribution(importance_samples, BOUNDS)
    
    
    # Step3-V3
    importance_samples, importance_density_f_x = MCMC(f_rbf, BOUNDS, N_SAMPLES // 5)
#    MCMC_PyMC3(f_rbf, BOUNDS, N_SAMPLES)
    
    if len(BOUNDS) == 2:
        plot_results(importance_samples, importance_density_f_x, function_name = "Density f_x", sampling_method = "Importance sampling", x_axis=0, y_axis=1)
        
        # Plot distribution histogram
        plot_3D_distribution(importance_samples, BOUNDS)
    
    # Compute f with new samples
    final_sampling = np.concatenate((reduced_uniform_sampling, importance_samples), axis=0)
    importance_f_x = []
    for x in importance_samples:
        importance_f_x.append(f(x, function_name = FUNCTION_NAME))
    final_estimated_f_x = estimated_f_x + importance_f_x
    
    if len(BOUNDS) == 2:
        plot_results(final_sampling, final_estimated_f_x, function_name = FUNCTION_NAME, sampling_method = SAMPLING_METHOD, x_axis=0, y_axis=1)
        
        # Plot distribution histogram
        plot_3D_distribution(importance_samples, BOUNDS)
        
    
    
def f(x, function_name = "ShiftedAckley"):
    """ 
    Parameters
    ----------
    x: list of float
        Sample
    function_name: str
        Name of the function used for computation

    Output
    ----------
    f(x): float
     
    """
    
    if function_name == "ShiftedAckley":
        # The Ackley function is widely used for testing optimization algorithms
        n = len(x) # Dimension of samples
        squared_sum = sum(map(lambda e:e*e,x))
        cos_sum = sum(map(lambda e:np.cos(2*np.pi*e),x))
        bias = 0
        return -20.0 * np.exp(-0.2 * np.sqrt((1/n)  * squared_sum))- np.exp( (1/n) * cos_sum) + np.e + 20 + bias


def weight_according_to_neighbours(samples, f_x, k = 10):
    """  Compute weights according to neighbours
    
    Parameters
    ----------
    samples: list of list of float
        Features for each sample
    f_x: list of float
        f(samples) values
    k: int
        Nummber of nearest neighboors to consider for each sample for computing weights
    ----------   
    Output
    ----------
    weights: list of float
        Computed weight for each sample
     
    """
    
    # Compute K nearest neighboors
    X = np.array(samples)
    N = len(samples)
    kdt = KDTree(X, leaf_size=50, metric='euclidean')
    k_neighboors = kdt.query(X, k=k, return_distance=False)
    
    # Compute weights
    weights = [0] * N
    for i in range(N):
        for j in range(len(k_neighboors[i])):
            weights[i] += (f_x[i] - f_x[j])**2
        weights[i] = np.sqrt(weights[i]) / N  
        
    # Normalize weights
    weights_sum = np.sum(weights)
    weights = weights / weights_sum
        
    return weights
    

def interpolate_from_samples(samples, f_x, method = "RBF"):
    """  Compute a continuous function from a set of discrete observations
    
    Parameters
    ----------
    samples: list of list of float
        Features for each sample
    f_x: list of float
        f(samples) values
    method: str
        Method used for interpolating the function
    ----------   
    Output
    ----------
    f: scipy.inteprolate function
        Interpolated continuous function
     
    """
    
    if method == "RBF":
        from scipy.interpolate import Rbf
        a = []
        for dim in range(len(samples[0])):
            a.append([samples[k][dim] for k in range(len(samples))])
        a.append(f_x)
        f = Rbf(*a, epsilon=1, smooth = 2)
        return f
    
def integrate(f, bounds):
    """  Compute the n_fold integration of a continuous function
    
    Parameters
    ----------
    f: scipy function
        Continuous function to integrate
    bounds: list list of floats
        Bounds for each input dimension
    ----------   
    Output
    ----------
    area: float
        Area under the curve
     
    """
    return nquad(f, bounds)
    
    
def ecdf(bounds, f, n_per_dim):
    """  Compute the multivariate empirical distribution function
    i.e. the fraction of observations with values smaller than the point.
    
    Two nice links that helped me building this function:
    # https://stats.stackexchange.com/questions/226935/algorithms-for-computing-multivariate-empirical-distribution-function-ecdf
    # https://silo.tips/download/a-problem-in-multivariate-statistics-algorithm-data-structure-and-applications
    
    Parameters
    ----------
    bounds: list list of floats
        Bounds for each input dimension
    f: scipy function
        Continuous function defined on bounds for computing the ecdf
    n_per_dim: int
        Number of sample per dimension for grid sampling
    ----------   
    Output
    ----------
    P: numpy array of floats
        The joint cumulative distribution function 
    edges: numpy array of floats
        Grid sample values taken by each random variable 
    """
    
    # Compute a grid of samples using interpolated f
    a = []
    for k in range(len(bounds)):
        a.append(np.linspace(bounds[k][0], bounds[k][1], n_per_dim))
    [*samples_by_dim] = np.meshgrid(*a)   
    samples = np.reshape(np.meshgrid(*a), (len(bounds), -1)).T
#    print("Samples:", samples)
    
    a_test = []
    for dim in range(len(samples[0])):
        a_test.append([samples[k][dim] for k in range(len(samples))])
    f_test = f(*a_test)  
#    print("f_test:", f_test)
    
    # Normalize before computing the joint probability mass function
    f_test_sum = np.sum(f_test)
    f_normalized = f_test / f_test_sum
#    print("f_normalized:", f_normalized)
#    print("Sum probas:", sum(f_normalized))
        
    # Rasterize the data points in a data grid for computign a normalized multivariate histogram
    P, edges = np.histogramdd(np.array(samples), bins = (n_per_dim,n_per_dim), weights = f_normalized)
#    print("P:",P)
#    print("edges:", edges)
    
    # Integrate in each dimension
    #                                Y = y1   Y = y                       Y <= y1   Y <= y
    # Going from a table    X = x1    a         b        to      X <= x1     a        a+b       
    #                       X = x2    c         d                X <= x2     c        c+d
    for dim in range(len(samples[0])):
        P = np.cumsum(P, axis=dim)
#        print("P:", P)
    
    return P, edges
    
    

def inverse_map_cdf(n_samples, bounds, sorted_proba_dict, kdt):
    """  Inverse map an uniform samplign using a computed CDF
        Issue encoutnered: it is not possible to inverse map a multidimensional function
        
        It may work by using joint cumulative densities if the different variables are independants:
        Compute the cdf of 1d marginal, pick an uniform random position just like the inverse transform sampling method; then, do this again on the 2nd dimension by conditioning on the already sampled position from the previous step
    Parameters
    ----------
    n_sample: int
        Number of samples to compute
    bounds: list list of floats
        Bounds for each input dimension
    sorted_proba_dict: dict of list of lsit of floats
        Dicitonnary which keys are eCDF values in ascending order
        Values are a list of corresponding coordinate
    kdt: scipy KDTree
        Structure for quickly quiering inverse mapping in the CDF
    ----------   
    Output
    ----------
    cdf_samples: list of list of float
        Importance sampling based on provided CDF.
    """
    
    # Sample an uniformly distributed sequence in the interval [0,1]
    uniform_sampling = sample([[0, 1]], method = SAMPLING_METHOD, n_samples = n_samples) 
    
    # Query closest probabilities for each sample
    distances, indices = kdt.query(uniform_sampling, 1) 
    
    # Inverse map by getting one of the coordinate corresponding to a strictly superior probability
    importance_samples = []
    sorted_proba = list(sorted_proba_dict.keys())
    for i in range(len(indices)):
        if (uniform_sampling[i][0] <= sorted_proba[indices[i][0]]) or i == len(indices) - 1:
            importance_samples.append(random.choice(sorted_proba_dict[sorted_proba[indices[i][0]]]))
        else:
            importance_samples.append(random.choice(sorted_proba_dict[sorted_proba[indices[i+1][0]]]))
    
    return importance_samples
        

    
#def MCMC_PyMC(density_f, bounds, n_samples):
#    """ Markov Chain Monte Carlo
#    
#    Nice explaination: https://pymcmc.readthedocs.io/en/latest/theory.html
#    Nice tutorial: https://pymcmc.readthedocs.io/en/latest/modelfitting.html
#    
#    The considered function is a blackbox. It is then impossible to directly use automatic differentiation.
#    How to manage blackbox?: 
#    https://docs.pymc.io/en/v3/pymc-examples/examples/case_studies/blackbox_external_likelihood.html
#
#    Parameters
#    ----------
#    samples: list of lists of floats
#        Samples used forcomputing f_x
#    f_x: list of floats
#        Images of samples through function f
#    function: str
#        Name of function
#    sampling_method: str
#        Name of the samplign method used to obtain provided samples
#    x_axis: int
#        Sampling space dimension to plot on x_axis
#    y_axis: int
#        Sampling space dimension to plot on y_axis
#
#    """
#    import theano.tensor as tt
#    import aesara.tensor as at
#    import pymc3 as pm
##    import seaborn as sbn
#    
#    input_size = 2
#    input_test = [0] * input_size
#    
#    def f():
#        def f_density(X):
##            f_X =  density_f(*X)
#            f_X = np.asarray(density_f(X[0], X[1]))
#            return f_X
#        return f_density
#          
##    X = sample(BOUNDS, method = SAMPLING_METHOD, n_samples = n_samples)
##    f_X = f(X, density_f)
#        
#    
#    # A funtion able to take Theano tensor objects, but internally cast them as floating point values that can be passed to the f function
#    # define a aesara Op for our likelihood function
#    class LogLike(at.Op):
#    
#        """
#        From https://colab.research.google.com/github/pymc-devs/pymc-examples/blob/main/examples/case_studies/blackbox_external_likelihood_numpy.ipynb#scrollTo=OprqgmlTNQfn
#        Specify what type of object will be passed and returned to the Op when it is
#        called. In our case we will be passing it a vector of values (the parameters
#        that define our model) and returning a single "scalar" value (the
#        log-likelihood)
#        """
#    
#        itypes = [at.dvector]  # expects a vector of parameter values when called
#        otypes = [at.dscalar]  # outputs a single scalar value (the log likelihood)
#    
#        def __init__(self, f, x):
#            """
#            Initialise the Op with various things that our log-likelihood function
#            requires. Below are the things that are needed in this particular
#            example.
#    
#            Parameters
#            ----------
#            f:
#                The f function we've defined
#            x:
#                The dependent variable (aka 'x') that our model requires
#            sigma:
#                The noise standard deviation that our function requires.
#            """
#    
#            # add inputs as class attributes
#            self.likelihood = f
#            self.x = x
#    
#        def perform(self, node, inputs, outputs):
#            # the method that is used when calling the Op
#            (theta,) = inputs  # this will contain my variables
#    
#            # call the log-likelihood function
#            logl = self.likelihood(theta, self.x)
#    
#            outputs[0][0] = np.array(logl)  # output the log-likelihood
#
#
#
#    # create our Op
#    x = np.linspace(-30, 30, 10)
#    logl = LogLike(f, data, x, sigma)    
#    
#        
#    with pm.Model() as model:
#        lim = 30
#        x0 = pm.Uniform('x0', -lim, lim)
#        x1 = pm.Uniform('x1', -lim, lim)
#        X = at.as_tensor_variable([x0,x1])
#        b = pm.Potential("likelihood", logl(X))
##        b = pm.DensityDist('f', f(), shape=input_size, testval=input_test)
#        step = pm.Metropolis()
#        trace = pm.sample(1000, step=step, cores=1)
#    
#    
#    sbn.kdeplot(trace['x0'][500::100])
#    sbn.kdeplot(trace['x1'][500::100])
#    plt.show()




def MCMC(density_f, bounds, n_samples, mode = "Metropolis"):
    """ Markov Chain Monte Carlo using Metropolis Hasting algorithm
    The algorithm iteratively sample candidates only from the last sampled candidate. 
    With some probability depending of the density function, the candidate is either accepted or rejected.
    
    Nice explanation: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
                      
    
    To improve the behaviour of the method regarding correlation of the samples, a good workaround would be to implement Gibbs sampling as explained here:
        https://en.wikipedia.org/wiki/Gibbs_sampling

    Parameters
    ----------
    density_f: scipy interpolate function
        The density function to sample from. It is not required for this density fucntio nto be exactly equal to a density function but only proportional to it.
    
    samples: list of lists of floats
        Samples used forcomputing f_x
    f_x: list of floats
        Images of samples through function f
    function: str
        Name of function
    sampling_method: str
        Name of the samplign method used to obtain provided samples
    x_axis: int
        Sampling space dimension to plot on x_axis
    y_axis: int
        Sampling space dimension to plot on y_axis

    """
    
    def posterior(x):
        # Return f(x) if x fits in bounds
        for k in range(len(x)):
            if x[k] < bounds[k][0]:
                return 0
            if x[k] > bounds[k][1]:
                return 0
        return density_f(*x)
    
    N = 100 * n_samples # Maximum number of tested samples
    s = 10
    n_sampled = 0
    samples = []
    f_X = []
    
    if mode == "Metropolis":    
        ### Initialization
        # An arbitrary point is chosen to be the first observation
        x = np.array([random.uniform(bounds[k][0], bounds[k][1]) for k in range(len(bounds))])
        p = posterior(x)
        # Random walk parameter: Arbitrary probability density suggesting a candidate for the next sample
        def jumping_distribution(x):
            xn = x + np.random.normal(size=len(bounds))
            return xn
        
        ### Main loop      
        i = 0
        while i <= N and n_sampled <= n_samples:
            # Generate a candidate for the next sample from the jumping distribution
            xn = jumping_distribution(x)
            # Compute the acceptance ratio
            pn = posterior(xn)
            alpha = pn / p
            # Accept or reject the candidate
            if alpha >= 1:
                p = pn
                x = xn
            else:
                u = np.random.rand()
                if u < alpha:
                    p = pn
                    x = xn
            if i % s == 0:
                n_sampled += 1
                samples.append(x)
                f_X.append(posterior(x))
            i += 1
    elif mode == "Gibbs":
        
        
    
    
    return samples, f_X
             
    
    



def plot_results(samples, f_x, function_name, sampling_method, x_axis=0, y_axis=1):
    """ Project sampling on 2 axes and plot the distribution

    Parameters
    ----------
    samples: list of lists of floats
        Samples used forcomputing f_x
    f_x: list of floats
        Images of samples through function f
    function: str
        Name of function
    sampling_method: str
        Name of the samplign method used to obtain provided samples
    x_axis: int
        Sampling space dimension to plot on x_axis
    y_axis: int
        Sampling space dimension to plot on y_axis

    """
    
    # Get samples components
    x = [samples[k][x_axis] for k in range(len(samples))]
    y = [samples[k][y_axis] for k in range(len(samples))]

    # Init figure
    fig = plt.figure()
    axs = []
    
    # Plot samples
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(x, y, marker="x", s = 0.5)
    ax.set_title("Projected sampling")
    axs.append(ax)
    
    # Plot function in 3D
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(x, y, f_x, marker="x", s = 0.5)
    ax.set_title("3D results")
    axs.append(ax)
    
    # Plot samples weighted by effect
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(x, y, c = f_x, s = 0.5, marker="x", cmap="plasma")
    ax.set_title("Projected sampling weighted by function value")
    axs.append(ax)
            
    for ax in axs:
        ax.set(xlabel="Projection on axis " + str(x_axis), ylabel="Projection on axis " + str(y_axis))

    for ax in axs:
        ax.label_outer()
    
    fig.suptitle(function_name + " function computed from " + sampling_method + " sampling method with " + str(len(samples)) + " samples of dim " + str(len(samples[0])))
    plt.show()
    
    
def plot_3D_distribution(samples, bounds):
    
    # Plot histograms of the distribution 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = [samples[k][0] for k in range(len(samples))]
    ys = [samples[k][1] for k in range(len(samples))]
    hist, xedges, yedges = np.histogram2d(xs, ys, bins=(15,15), range=bounds)
      
    # Construct arrays for the anchor positions of the bars.
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:]) - abs(xedges[1]-xedges[0])
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)
    
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    
    cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title("3D distribution histogram")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
    
        
def density_function(method = "1DGaussian", plot = True):        
    if method == "1DGaussian":   
        def gaussian(mu=0, sigma=1):
            distribution = stats.norm(mu, sigma)
            return distribution
        distribution = gaussian(mu=0, sigma=0.5) 
        return distribution
        

                   
def plot_density_function(xs, ys, xmin=0, xmax=1, ymax=1.2, method = "1DGaussian"):    
    plt.plot(xs, ys, label= method + " Density Function") 
    plt.fill_between(xs, ys, 0, alpha=0.2)
    plt.xlim(xmin, xmax) 
    plt.ylim(0, ymax)
    plt.xlabel("x") 
    plt.ylabel("y") 
    plt.legend()
    plt.show()

    
def batch_sample(f, num_samples, xmin=0, xmax=1, ymax=1.2, batch=1000, method = "1DGaussian"):
    samples = []
    while len(samples) < num_samples:
        x = np.random.uniform(low=xmin, high=xmax, size=batch)
        y = np.random.uniform(low=0, high=ymax, size=batch)
        if method == "1DGaussian":
            samples += x[y < f.pdf(x)].tolist()
        else:
            samples += x[y < f(x)].tolist()
    return samples[:num_samples]

    
def plot_samples(xs, ys, samples, xmin=0, xmax=1, ymax=1.2, method = "1DGaussian"):
    plt.plot(xs, ys, label= method + " Density Function" )
    plt.hist(samples, density=True, alpha=0.2, label="Sample distribution with " + method)
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)
    plt.xlabel("x") 
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
    
        
if __name__ == "__main__":
    main()    
    
    
    