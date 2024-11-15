import numpy as np
import os
import pickle
import tqdm
from joblib import Parallel, delayed

def bayesian_bootstrap(data, n_bootstrap=10000, credibility_interval=0.95):
    """
    Perform Bayesian bootstrapping on the input data and compute credibility intervals.

    Parameters:
    - data: 1D numpy array of rate constant estimates at a particular time point.
    - n_bootstrap: Number of bootstrap iterations (default 10,000).
    - credibility_interval: Desired credibility interval (default 95%).

    Returns:
    - mean_estimates: Mean of the bootstrap samples.
    - lower_bound: Lower bound of the credibility interval.
    - upper_bound: Upper bound of the credibility interval.
    """
    n_samples = data.shape[0]
    
    # Initialize storage for bootstrap estimates
    bootstrap_means = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Draw random weights for the empirical data (dirichlet distribution is equivalent to resampling multinomial weights)
        weights = np.random.dirichlet(np.ones(n_samples))
        # Compute weighted mean
        bootstrap_means[i] = np.sum(weights * data)
    
    # Compute mean of bootstrap estimates
    mean_estimates = np.mean(bootstrap_means)
    
    # Calculate credibility interval bounds
    lower_bound = np.percentile(bootstrap_means, (1 - credibility_interval) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + credibility_interval) / 2 * 100)
    
    return mean_estimates, lower_bound, upper_bound

def compute_bayesian_bootstrap_intervals(data, n_bootstrap=10000, credibility_interval=0.95, n_jobs=-1):
    """
    Compute Bayesian bootstrap credibility intervals for each time point in parallel.

    Parameters:
    - data: 2D numpy array of shape (N_traj, N_time_points), where N_traj is the number of trajectories
            and N_time_points is the number of time points.
    - n_bootstrap: Number of bootstrap iterations (default 10,000).
    - credibility_interval: Desired credibility interval (default 95%).
    - n_jobs: Number of CPU cores to use. -1 means using all available cores.

    Returns:
    - mean_estimates: 1D numpy array of mean estimates at each time point.
    - lower_bounds: 1D numpy array of lower bounds at each time point.
    - upper_bounds: 1D numpy array of upper bounds at each time point.
    """
    N_time_points = data.shape[1]

    # Define a helper function for processing a single time point
    def process_time_point(t):
        return bayesian_bootstrap(
            data[:, t], n_bootstrap=n_bootstrap, credibility_interval=credibility_interval
        )

    # Use joblib.Parallel to parallelize the computation over time points
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_time_point)(t) for t in tqdm.tqdm(range(N_time_points))
    )

    # Unpack the results
    mean_estimates, lower_bounds, upper_bounds = map(np.array, zip(*results))

    return mean_estimates, lower_bounds, upper_bounds

def load_probability_flows(project_folder, params, replicates=None):
    """
    Load probability flow data from each replicate in the experiment folder.

    Parameters:
    - experiment_folder: Path to the experiment folder.
    - params: Dictionary of parameters used in the experiment.
    - replicates: List of replicate IDs to load. If None, all replicates are loaded.

    Returns:
    - probability_flows: 3D numpy array of shape (N_replicates, N_time_points, N_bins).
    """
    experiment_folder = os.path.join(project_folder, params['experiment_name'])

    if replicates is None:
        replicates = range(params['n_replicates'])
    
    # Loop through replicates and load recycled weight data
    recycled_weight = []
    for rep in replicates:
        with open(os.path.join(experiment_folder,f'{rep:03d}','recycled_weight.pkl'), 'rb') as f:
            recycled_weight.append(pickle.load(f))
    recycled_weight = np.array(recycled_weight)

    # Compute cumulative sum of recycled weight for each replicate
    # Divide by the iteration number to get the average probability flow per iteration
    recycled_weight_cumsum = []
    time_points = np.arange(1,params['n_iters']+1) * params['n_steps_per_iter']
    for rep in replicates:
        data = np.cumsum(recycled_weight[rep])
        recycled_weight_cumsum.append(data/time_points)
    recycled_weight_cumsum = np.array(recycled_weight_cumsum)
    
    return time_points, recycled_weight_cumsum


# Computing minimal distance from walker point cloud to each saddle point
def compute_saddle_point_mindist(args):
    project_dir, replicate_id, saddle_points = args
    replicate_dir = os.path.join(project_dir, replicate_id)
    with open(os.path.join(replicate_dir, 'trajs.pkl'), 'rb') as f:
        trajs = pickle.load(f)
    
    n_iters = len(trajs)
    n_steps_per_iter = trajs[0].shape[1]
    n_saddle_points = saddle_points.shape[0]

    saddle_point_mindists = np.empty((n_saddle_points, n_iters, n_steps_per_iter))
    for iter_idx, traj in enumerate(trajs):
        for timestep_idx in range(n_steps_per_iter):
            walker_positions = traj[:, timestep_idx, :]
            # Compute the minimal distance all walkers and each saddle point
            dists = np.linalg.norm(walker_positions[:, None, :] - saddle_points[None, :, :], axis=2)
            saddle_point_mindists[:,iter_idx, timestep_idx] = np.min(dists, axis=0)
    return saddle_point_mindists