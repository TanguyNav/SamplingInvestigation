# -*- coding: utf-8 -*-
""" A script for investigating sampling methods behaviour in high dimensional space"""

__authors__ = "tnavez"
__contact__ = "tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 05 2022"

SAMPLING_METHOD = ["Random", "Grid", "LHS", "OptimizedLHS", "Halton", "ScrambledHalton", "CVT_RandomStart", "CVT_HaltonStart", "LCVT"] 
BOUNDS = [[0, 10] for i in range(9)]
N_SAMPLES = 1050

# Libraries
import numpy as np
from scipy.stats import qmc, pearsonr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import itertools

# Main function
def main(args=None):    
    
    # Sample
    samples_by_method = {}
    for method in SAMPLING_METHOD:               
        samples_by_method[method] = sample(BOUNDS, method=method, n_samples=N_SAMPLES)
        
    # Plot
    plot_projected_samples(samples_by_method, BOUNDS, x_axis=2, y_axis=7)
            
      
    
def sample(bounds, method = "Random", n_samples = 1500):
    """ Function to manage sampling of data point from a given distribution

    Parameters
    ----------
    bounds: list of lists
        Bounds for sampling intervals
    method: str in {Random, Grid, Halton}
        Method for data sampling.
        Random: sample points at random.
        Grid: exhaustively sample input data space.
        LHS: Latin Hypercube Sampling generates a set of samples such that each sample point is formed by randomly selecting one coordinate value from a grid from each dimension
        OptimizedLHS: Modify an initial Latin Hypercube Sampling to optimzie a quality metrics for improving coverage of overalld esign space while keeping good coverage along each dimension individually
        Halton: low discrepancy sequence
        ScrambledHalton: low discrepancy sequence with scrambling to ameliorate Halton sequence by limitating severe striping artifacts
        CVT_RandomStart: Centroidal Voronoi Tessellation generating sample points located at the center of mass of each Voronoi cell covering the input space, start from a Random distribution
        CVT_HaltonStart: Centroidal Voronoi Tessellation generating sample points located at the center of mass of each Voronoi cell covering the input space, start from an Halton distribution
        LCVT: Latinized Centroidal Voronoi Tessellation as described in "Smart sampling and incremental function learning for very large high dimensional data"
    n_samples: int
        Number of samples. This input number is updated to obtain an homogeneous Grid strategy.

    Output
    ----------
    samples: list of lists of float
        Samples obtaiend usign the provided method
    """
    
    if method == "Random":
        print(">>> Start Random sampling ...")
        samples = [0] * n_samples
        for i in range(n_samples):
            samples[i] = [np.random.uniform(bounds[j][0],bounds[j][1]) for j in range(len(bounds))]        
        print("... End <<<")
        
    elif method == "Grid":
        print(">>> Start Grid sampling ...")
        n_sample_by_interval = max(2, int(np.exp(np.log(n_samples)/len(bounds))))
        all_sampling = []
        for j in range(len(bounds)):           
            sampling_j = list(np.linspace(bounds[j][0],bounds[j][1], n_sample_by_interval))   
            all_sampling.append(sampling_j)
        samples = [list(x) for x in np.array(np.meshgrid(*all_sampling)).T.reshape(-1,len(all_sampling))]
        print("... End <<<")
        
    elif method == "LHS":
        print(">>> Start LHS sampling ...")
        sampler = qmc.LatinHypercube(d=len(bounds))
        samples_01 = sampler.random(n=n_samples)
        l_bounds = [bounds[j][0] for j in range(len(bounds))]
        u_bounds = [bounds[j][1] for j in range(len(bounds))]
        samples = qmc.scale(samples_01, l_bounds, u_bounds) # Scale samples to bounds
        print("... End <<<")
        
    elif method == "OptimizedLHS":
        print(">>> Start OptimizedLHS sampling ...")
        sampler = qmc.LatinHypercube(d=len(bounds), optimization="random-cd")
        samples_01 = sampler.random(n=n_samples)
        l_bounds = [bounds[j][0] for j in range(len(bounds))]
        u_bounds = [bounds[j][1] for j in range(len(bounds))]
        samples = qmc.scale(samples_01, l_bounds, u_bounds) # Scale samples to bounds
        print("... End <<<")
        
    elif method == "Halton":
        print(">>> Start Halton sampling ...")
        sampler = qmc.Halton(d=len(bounds), scramble=False)
        samples_01 = sampler.random(n=n_samples)
        l_bounds = [bounds[j][0] for j in range(len(bounds))]
        u_bounds = [bounds[j][1] for j in range(len(bounds))]
        samples = qmc.scale(samples_01, l_bounds, u_bounds) # Scale samples to bounds
        print("... End <<<")
        
    elif method == "ScrambledHalton":
        print(">>> Start ScrambledHalton sampling ...")
        sampler = qmc.Halton(d=len(bounds), scramble=True)
        samples_01 = sampler.random(n=n_samples)
        l_bounds = [bounds[j][0] for j in range(len(bounds))]
        u_bounds = [bounds[j][1] for j in range(len(bounds))]
        samples = qmc.scale(samples_01, l_bounds, u_bounds) # Scale samples to bounds
        print("... End <<<")
        
    elif method == "CVT_RandomStart":  
        print(">>> Start CVT_RandomStart sampling ...")
        # Generate enough random points to cover the domain densely using Random sampling
        init_n_samples = 100 * n_samples
        init_samples = [0] * init_n_samples
        for i in range(init_n_samples):
            init_samples[i] = [np.random.uniform(bounds[j][0],bounds[j][1]) for j in range(len(bounds))]    
        
        # Identify the n_samples generators of the initial distribution using Kmeans for computing centroids
        kmeans = KMeans(
            init="k-means++",
            algorithm="full", #lloyd
            n_clusters=n_samples,
            n_init=1,
            max_iter=100,
            tol=0.0001,
            )
        kmeans.fit(init_samples)
        centroids = kmeans.cluster_centers_
        samples = centroids
        print("... End <<<")
        
    elif method == "CVT_HaltonStart":
        print(">>> Start CVT_HaltonStart sampling ...")
        # Generate enough random points to cover the domain densely using Halton sampling
        init_n_samples = 100 * n_samples
        sampler = qmc.Halton(d=len(bounds), scramble=True)
        samples_01 = sampler.random(n=init_n_samples)
        l_bounds = [bounds[j][0] for j in range(len(bounds))]
        u_bounds = [bounds[j][1] for j in range(len(bounds))]
        init_samples = qmc.scale(samples_01, l_bounds, u_bounds) # Scale samples to bounds
        
        # Identify the n_samples generators of the initial distribution using Kmeans for computing centroids
        kmeans = KMeans(
            init="k-means++",
            algorithm="full", #lloyd
            n_clusters=n_samples,
            n_init=1,
            max_iter=100,
            tol=0.0001,
            )
        kmeans.fit(init_samples)
        centroids = kmeans.cluster_centers_
        samples = centroids   
        print("... End <<<")

    elif method == "LCVT":
        print(">>> Start LCVT sampling ...")
        # Generate enough random points to cover the domain densely using Halton sampling
        init_n_samples = 100 * n_samples
        sampler = qmc.Halton(d=len(bounds), scramble=True)
        samples_01 = sampler.random(n=init_n_samples)
        l_bounds = [bounds[j][0] for j in range(len(bounds))]
        u_bounds = [bounds[j][1] for j in range(len(bounds))]
        init_samples = qmc.scale(samples_01, l_bounds, u_bounds) # Scale samples to bounds
        
        # Identify the n_samples generators of the initial distribution using Kmeans for computing centroids
        kmeans = KMeans(
            init="k-means++",
            algorithm="full", #lloyd
            n_clusters=n_samples,
            n_init=1,
            max_iter=100,
            tol=0.0001,
            )
        kmeans.fit(init_samples)
        centroids = kmeans.cluster_centers_
        samples = centroids.tolist()
        
        # Latinize the distribution
        for j in range(len(bounds)): 
            # Reorder samples in increasing order regarding dimension j
            samples.sort(key=lambda x:int(x[j]))
            # Divide the range of values in the j-th dimension into n_samples equispaced bins
            bin_length = (bounds[j][1] - bounds[j][0]) / n_samples
            assiocated_bins_j = [samples[k][j] // bin_length for k in range(n_samples)]
            # Shift non self-contained data in their bin to a random position position in the correct bin
            for i in range(n_samples):
                if assiocated_bins_j[i] != i:
                    samples[i][j] = np.random.uniform(i*bin_length, (i+1)*bin_length)
        print("... End <<<")
        
    return samples
    
    
def plot_projected_samples(samples_by_method, bounds, x_axis=2, y_axis=7):
    """ Project sampling on 2 axes and plot the distribution

    Parameters
    ----------
    samples_by_method: dict of list of lists of floats
        Sampels obtained for different strategy
    bounds: list of lists
        Bounds for sampling intervals
    x_axis: int
        Sampling space dimension to plot on x_axis
    y_axis: int
        Sampling space dimension to plot on y_axis

    """
    methods = list(samples_by_method.keys())
    n_methods = len(methods)
    fig, axs = plt.subplots(n_methods // 3 + 1, 3, squeeze=False)
    for i in range(n_methods // 3 + 1):
        for j in range(3):
            if (i*3 + j) < n_methods:
                x_projected = [samples_by_method[methods[i*3+j]][k][x_axis] for k in range(len(samples_by_method[methods[i*3+j]]))]
                y_projected = [samples_by_method[methods[i*3+j]][k][y_axis] for k in range(len(samples_by_method[methods[i*3+j]]))]
                axs[i,j].scatter(x_projected, y_projected, marker="x", s = 0.8)
                
                # Rescale data in [0,1] for computign assessment metrics
                l_bounds = [bounds[j][0] for j in range(len(bounds))]
                u_bounds = [bounds[j][1] for j in range(len(bounds))]
                samples_by_method_01 = qmc.scale(samples_by_method[methods[i*3+j]], l_bounds, u_bounds, reverse=True)
                
                # Compute and display the maximum correlation score between input variables as sugested in https://waterprogramming.wordpress.com/2018/06/11/evaluating-and-visualizing-sampling-quality/
                # High correlation between sample values means there will be larger variations of the output values
                values_per_dim = []
                for i_dim in range(len(bounds)):
                    values_per_dim.append([samples_by_method_01[k][i_dim] for k in range(len(samples_by_method_01))])
                max_pearson = 0
                for i_dim in range(len(bounds)):
                    for j_dim in range(i_dim+1, len(bounds)):
                        pearson_ij = abs(pearsonr(values_per_dim[i_dim], values_per_dim[j_dim]).statistic)
                        if pearson_ij > max_pearson:
                            max_pearson = pearson_ij
                    
                # Compute and display the discrepancy score
                # The discrepancy is a uniformity criterion used to assess the space filling of a number of samples in a hypercube. 
                # The lower the value, the better the coverage of the space is.             
                discrepancy_score = qmc.discrepancy(samples_by_method_01, method="CD") #MD

                axs[i,j].set_title(methods[i*3+j] + " sampling: c=" + "{:.2e}".format(max_pearson) + ", d=" + "{:.2e}".format(discrepancy_score))
    
    for ax in axs.flat:
        ax.set(xlabel="Projection on axis " + str(x_axis), ylabel="Projection on axis " + str(y_axis))

    for ax in axs.flat:
        ax.label_outer()
    
    fig.suptitle("Comparison of sampling methods for " + str(N_SAMPLES) + " samples and " + str(len(bounds)) + " dimensions.")
    plt.show()
    
    
    
    
if __name__ == "__main__":
    main()    
    
    
    