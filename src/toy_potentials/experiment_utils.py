import numpy as np

def initialize_walkers(num_walkers, x_loc=-1, y_loc=-1, x_scale=0.1, y_scale=0.1, tot_weight=1.0):
    walkers = np.zeros((num_walkers, 3))  # Columns for x, y, and weight
    walkers[:, 0] = np.random.normal(loc=x_loc, scale=x_scale, size=num_walkers)
    walkers[:, 1] = np.random.normal(loc=y_loc, scale=y_scale, size=num_walkers)
    walkers[:, 2] = tot_weight / num_walkers  # Assign equal weights
    return walkers

# Move walkers based on the energy landscape
def move_walkers_iteration(model, walkers_coords, n_steps_iter=100, n_threads=8):
    walker_trajs = model.trajectory(walkers_coords, n_steps_iter, n_jobs=n_threads)
    return walker_trajs

def simple_we_resample(iter_final_coords, iter_weights, iter_bin_ids, n_walkers_per_bin, min_weight):
    """
    Simple weighted ensemble resampling algorithm that resamples walkers in each bin based on their weights.
    The resampling is done to ensure that each bin has exactly n_walkers_per_bin walkers.

    Parameters
    ----------
    iter_final_coords : np.ndarray
        The final coordinates of the walkers in the current iteration.
    iter_weights : np.ndarray
        The weights of the walkers in the current iteration.
    iter_bin_ids : np.ndarray
        The bin IDs of the walkers in the current iteration.
    n_walkers_per_bin : int
        The number of walkers per bin after resampling.
    min_weight : float
        The minimum weight that each walker should have after resampling.
    """
    # Resample to ensure each populated bin has exactly n_walkers_per_bin
    updated_iter_coords = []
    updated_iter_weights = []
    updated_iter_bin_ids = []
    for bin_id in np.unique(iter_bin_ids):
        # Walkers in current bin
        walkers_in_bin = np.where(iter_bin_ids == bin_id)[0]
        # Compute the total weight of the walkers in this bin
        weight_in_bin = iter_weights[walkers_in_bin].sum()
        # Resample existing walkers based on their weights up to the prescribed number
        if weight_in_bin/n_walkers_per_bin >= min_weight:
            new_walkers = np.random.choice(walkers_in_bin, 
                n_walkers_per_bin, 
                p=iter_weights[walkers_in_bin]/np.sum(iter_weights[walkers_in_bin]), 
                replace=True)
            # Distribute the weight equally among the resampled walkers
            updated_weight = np.ones(n_walkers_per_bin) * weight_in_bin / n_walkers_per_bin
            assert np.isclose(updated_weight.sum(), weight_in_bin)
            updated_iter_weights.append(updated_weight)
            # Update walker positions by keeping only the kept walkers
            updated_iter_coords.append(iter_final_coords[new_walkers])
            # Bin IDs are updated to reflect the new bin assignments
            updated_iter_bin_ids.append(np.ones(n_walkers_per_bin) * bin_id)
        else:
            # Dont resample if it would drop the per-walker weight too low
            updated_iter_weights.append(iter_weights[walkers_in_bin])
            updated_iter_coords.append(iter_final_coords[walkers_in_bin])
            updated_iter_bin_ids.append(iter_bin_ids[walkers_in_bin])
    # Combine the updated walker information
    updated_iter_coords = np.vstack(updated_iter_coords)
    updated_iter_weights = np.hstack(updated_iter_weights)
    updated_iter_bin_ids = np.hstack(updated_iter_bin_ids)
    assert np.isclose(updated_iter_weights.sum(), 1)
    assert np.all(updated_iter_weights >= min_weight)
    return updated_iter_coords, updated_iter_weights, updated_iter_bin_ids

def skip_resampling(iter_parent_coords, iter_weights, iter_bin_ids, n_walkers_per_bin, min_weight):
    return iter_parent_coords, iter_weights, iter_bin_ids