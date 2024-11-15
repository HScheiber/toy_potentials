import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from typing import Union

def furthest_point_sample(
    n_samples: int,
    sample_points: np.ndarray,
    initial_samples: Union[np.ndarray, None] = None,
    return_indices: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Sample from a point cloud (sample_points) and return a set of samples where each point
    is added sequentially at a point maximally distant from all other sampled points.
    The first point is selected randomly from the cloud, unless given by initial_samples.

    Args:
        n_samples (int): The number of samples to generate.
        sample_points (np.ndarray): The point cloud too sample from.
        initial_samples (np.ndarray): An initial set of samples to start with (not necessarily in the point cloud).
        return_indices (bool): If True, also return the indices of the selected points in the original point cloud.
    Returns:
        best_samples (np.ndarray): An (n_samples, n_dim) array of the best samples found.
    """
    n_sample_points = sample_points.shape[0]
    if n_samples > n_sample_points:
        raise ValueError(
            "n_samples cannot be greater than the number of points in sample_points."
        )

    if initial_samples is None:
        # Start with a random point from the point cloud
        first_idx = np.random.randint(n_sample_points)
        subset_indices = [first_idx]
        n_remaining = n_samples - 1
    else:
        # Combine the initial samples and the sample points into a point cloud
        sample_points = np.vstack((sample_points, initial_samples))
        n_sample_points = sample_points.shape[0]

        # Get indices of the initial samples within the combined point cloud
        subset_indices = np.arange(
            n_sample_points - initial_samples.shape[0], n_sample_points
        ).tolist()
        n_remaining = n_samples - len(subset_indices)

    # Initialize array to keep track of minimum distances to selected points
    min_distances = np.full(n_sample_points, np.inf)
    selected_mask = np.zeros(n_sample_points, dtype=bool)
    selected_mask[subset_indices] = True

    # Update min_distances for initial selected points
    if initial_samples is None:
        # Compute distances from all points to the first selected point
        first_point = sample_points[subset_indices[0]]
        diff = sample_points - first_point
        distances = np.linalg.norm(diff, axis=1)
        min_distances = distances
    else:
        # Compute distances from all points to each of the initial samples
        for idx in subset_indices:
            point = sample_points[idx]
            diff = sample_points - point
            distances = np.linalg.norm(diff, axis=1)
            min_distances = np.minimum(min_distances, distances)

    # Set distances for selected points to zero
    min_distances[selected_mask] = 0

    for _ in range(n_remaining):
        # Select the point with the maximum minimum distance
        selected_idx = np.argmax(min_distances)
        subset_indices.append(selected_idx)
        selected_mask[selected_idx] = True

        # Update the min_distances array with distances to the newly selected point
        new_point = sample_points[selected_idx]
        diff = sample_points - new_point
        distances = np.linalg.norm(diff, axis=1)
        min_distances = np.minimum(min_distances, distances)

        # Set distance for the newly selected point to zero
        min_distances[selected_idx] = 0

    # Prepare the output
    best_samples = sample_points[subset_indices]

    if return_indices:
        indices = np.array(subset_indices)
        return best_samples, indices
    else:
        return best_samples

def rectilinear_binning(iter_final_coords, **kwargs):
    # Set up 2D rectilinear binning within the specified bounds
    x_bounds = kwargs['x_bounds']
    y_bounds = kwargs['y_bounds']
    n_bins_x = kwargs['n_bins_x']
    n_bins_y = kwargs['n_bins_y']

    # Compute the bin edges
    x_edges = np.linspace(x_bounds[0], x_bounds[1], n_bins_x+1)
    y_edges = np.linspace(y_bounds[0], y_bounds[1], n_bins_y+1)

    # Assign each coord in iter_final_coords to a bin
    iter_bin_ids = np.zeros(iter_final_coords.shape[0], dtype=int)
    for i, coord in enumerate(iter_final_coords):
        x_bin = np.digitize(coord[0], x_edges) - 1
        y_bin = np.digitize(coord[1], y_edges) - 1
        iter_bin_ids[i] = x_bin + y_bin * n_bins_x
    
    return iter_bin_ids

def fpvt_binning(iter_final_coords, **kwargs):
    # This binning scheme simply samples furthest points from the set of iter_final_coords
    n_bins = kwargs['n_bins']

    # Perform FPS
    vor_generator_coords = furthest_point_sample(n_samples = n_bins, 
                        sample_points = iter_final_coords)
    
    # Assign each walker to its nearest Voronoi generator point
    tree = KDTree(vor_generator_coords)
    return tree.query(iter_final_coords)[1]

def string_binning(iter_final_coords, iter_weights, string_object, **kwargs):
    """
    Assigns walkers to bins based on the string method and updates the string.
    """
    # Run one step of the string method
    string_object.run_step(iter_final_coords, iter_weights)

    # The bin assignments are in string_object.bin_assignment
    iter_bin_ids = string_object.bin_assignment.astype(int)
    return iter_bin_ids

def load_binning(binning_strategy, **kwargs):
    """
    Load the binning function based on the specified strategy.
    """
    if binning_strategy.lower() == 'fpvt':
        def binning_function(iter_init_coords, iter_final_coords, iter_weights):
            iter_bin_ids = fpvt_binning(iter_final_coords, **kwargs)
            return iter_bin_ids
    elif binning_strategy.lower() == 'rectilinear':
        def binning_function(iter_init_coords, iter_final_coords, iter_weights):
            iter_bin_ids = rectilinear_binning(iter_final_coords, **kwargs)
            return iter_bin_ids
    elif binning_strategy.lower() == 'string':
        # Extract required parameters from kwargs
        start_state = kwargs['start_state']
        end_state = kwargs['end_state']
        n_bins = kwargs['n_bins']
        update_rate = kwargs.get('update_rate', 0.1)
        time_window = kwargs.get('time_window', 10)
        smoothing_type = kwargs.get('smoothing_type', 'elastic')
        smoothing_factor = kwargs.get('smoothing_factor', 0.1)
        num_basis_functions = kwargs.get('num_basis_functions', 3)

        # Create the string object
        string_object = WeightedEnsembleStringMethod(
            start_state=start_state,
            end_state=end_state,
            n_bins=n_bins,
            update_rate=update_rate,
            time_window=time_window,
            smoothing_type=smoothing_type,
            smoothing_factor=smoothing_factor,
            num_basis_functions=num_basis_functions
        )

        def binning_function(iter_init_coords, iter_final_coords, iter_weights):
            iter_bin_ids = string_binning(iter_final_coords, iter_weights, string_object, **kwargs)
            return iter_bin_ids, string_object.path
    else:
        raise ValueError(f'Unknown binning strategy: {binning_strategy}')
    return binning_function


class WeightedEnsembleStringMethod:
    def __init__(self, 
                 start_state, 
                 end_state, 
                 n_bins, 
                 update_rate=0.1, 
                 time_window=10,
                 smoothing_type='elastic',
                 smoothing_factor=0.1,
                 num_basis_functions=3):
        """
        Initialize the Weighted Ensemble String method.
        
        Parameters:
        - start_state: The starting point of the string (e.g., the reactant state).
        - end_state: The ending point of the string (e.g., the product state).
        - n_bins: The number of bins (string nodes) along the string.
        - update_rate: Relaxation parameter controlling how quickly the string moves toward the average.
        - time_window: Number of previous steps to consider for time averaging of walker positions.
        - smoothing_type: Type of smoothing to apply to the string ('elastic' or 'sinusoidal').
        - smoothing_factor: Controls how aggressively the string smoothing is applied (κ in the formula) if using elastic smoothing.
        - num_basis_functions: Number of sinusoidal basis functions to use for curve fitting if using sinusoidal smoothing.
        """
        self.n_bins = n_bins              # Number of Voronoi cells
        self.update_rate = update_rate    # Relaxation parameter ζ
        self.time_window = time_window    # Time window for averaging walker positions
        self.smoothing_type = smoothing_type   # Type of smoothing to apply
        self.smoothing_factor = smoothing_factor*n_bins*update_rate  # Smoothing parameter κ^n
        self.n_dim = len(start_state) # Dimensionality of the state space
        self.P = num_basis_functions      # Number of basis functions for curve fitting

        # Initialize string with linearly interpolated points between start_state and end_state
        self.start_state = np.asarray(start_state)
        self.end_state = np.asarray(end_state)
        self.path = self.initialize_string()

        # Initialize position history for time averaging after smoothing
        self.position_history = {i: [self.path[i]] for i in range(n_bins)}  # History of positions for time averaging
        self.bin_assignment = None  # Tracks which bin each walker belongs to

    def initialize_string(self):
        """
        Initializes the string by linearly interpolating between the start and end states.
        
        Parameters:
        - start_state: The starting point of the string (as a numpy array).
        - end_state: The ending point of the string (as a numpy array).
        
        Returns:
        - init_path: The initialized string as a numpy array of shape (n_bins, n_dimensions).
        """
        return np.linspace(self.start_state, self.end_state, self.n_bins)
    
    def update_bins(self, walker_positions):
        """
        Maps walkers to the nearest Voronoi seed points (string nodes) using KDTree.
        
        Parameters:
        - walker_positions: The positions of the walkers (shape: [n_replicas, n_dimensions]).
        
        Returns:
        - bin_indices: Indices of the closest bins (Voronoi seeds) for each walker.
        """
        # Build KD-Tree for the current path (seed points)
        tree = KDTree(self.path)
        _, self.bin_assignment = tree.query(walker_positions)

    def sinusoidal_basis(self, Lambda):
        """
        Generate sinusoidal basis functions for curve fitting parameterized by Lambda.
        """
        # basis = np.zeros((self.P, self.n_dim))
        # for i in range(self.n_dim):
        #     for j in range(1, self.P + 1):
        #         basis[j-1, i] = np.sin(j * np.pi * Lambda)
        return np.array([np.sin(j * np.pi * Lambda) for j in range(1, self.P+1)])

    def optimize_sigma_given_lambda(self, lambdas):
        """
        Optimize the sinusoidal coefficients σ_ij for a fixed set of lambdas.

        Parameters:
        lambdas : (M,) ndarray
            The parameterization values for the string points.
        Returns:
        sigma_ij : (P, N) ndarray
            Optimized sinusoidal coefficients for the curve fitting.
        """
        phi_star = np.copy(self.path)
        def objective_sigma(sigma_flat):
            sigma_ij = sigma_flat.reshape(self.P, self.n_dim)
            chi_squared = 0.0
            for alpha in range(1, self.n_bins - 1):  # Don't optimize endpoints (0 and M-1)
                t = lambdas[alpha]
                phi_cur_alpha = phi_star[0] + (phi_star[-1] - phi_star[0]) * t
                basis = self.sinusoidal_basis(t)
                phi_cur_alpha += np.sum(basis[:, np.newaxis] * sigma_ij, axis=0)
                chi_squared += np.sum((phi_cur_alpha - phi_star[alpha]) ** 2)
            return chi_squared

        initial_guess_sigma = np.zeros((self.P, self.n_dim)).flatten()
        result = minimize(objective_sigma, initial_guess_sigma, method='BFGS')
        return result.x.reshape(self.P, self.n_dim)

    def optimize_lambda_given_sigma(self, sigma_ij):
        """
        Optimize the lambdas for a fixed set of sinusoidal coefficients σ_ij independently.

        Parameters:
        phi_star : (M, N) ndarray
            Input array representing M string nodes in N-dimensional space.
        sigma_ij : (P, N) ndarray
            The sinusoidal coefficients for curve fitting.
        
        Returns:
        lambdas : (M,) ndarray
            Optimized parameterization values for the string points.
        """
        phi_star = np.copy(self.path)

        # Initialize lambdas with λ_0 = 0 and λ_{M-1} = 1
        lambdas = np.zeros(self.n_bins)
        lambdas[-1] = 1

        # Optimize each lambda independently
        chi_squared = 0.0
        for alpha in range(1, self.n_bins - 1):
            def objective_lambda(lambda_alpha):
                t = lambda_alpha[0]
                phi_cur_alpha = phi_star[0] + (phi_star[-1] - phi_star[0]) * t
                basis = self.sinusoidal_basis(t)
                phi_cur_alpha += np.sum(basis[:, np.newaxis] * sigma_ij, axis=0)
                return np.sum((phi_cur_alpha - phi_star[alpha]) ** 2)

            initial_guess_lambda = lambdas[alpha]  # Initial guess for lambda
            result = minimize(objective_lambda, initial_guess_lambda, bounds=[(0, 1)], method='L-BFGS-B')
            lambdas[alpha] = result.x
            chi_squared += result.fun
        return lambdas, chi_squared

    def smooth_curve_fit(self, max_iter=1000, tol=1e-1):
        """
        Perform multidimensional curve fitting using sinusoidal basis functions.
        
        Parameters:
        phi_star : (M, N) ndarray
            Input array representing M string nodes in N-dimensional space.
        max_iter : int, optional
            Maximum number of iterations for optimization.

        Returns:
        fitted_curve : (M, N) ndarray
            Fitted smooth curve with M points in N-dimensional space.
        """
        phi_star = np.copy(self.path)
        # Initial guess for λ between [0,1], fix λ_0 = 0 and λ_{M-1} = 1
        lambdas = np.linspace(0, 1, self.n_bins)
        sigma_ij = np.zeros((self.P, self.n_dim))

        chi_squared_current = np.inf
        for _ in range(max_iter):
            # Step 1: Optimize sigma given lambda
            sigma_ij = self.optimize_sigma_given_lambda(lambdas)

            # Step 2: Optimize each lambda independently given sigma
            lambdas, chi_squared_new = self.optimize_lambda_given_sigma(sigma_ij)

            # Check for convergence
            if np.abs(chi_squared_new - chi_squared_current) < tol:
                break

        # Generate the string nodes fitted to the curve
        phi_star_cur = np.zeros((self.n_bins, self.n_dim))
        for alpha in range(self.n_bins):
            Lambda = lambdas[alpha]
            phi_star_cur[alpha] = phi_star[0] + (phi_star[-1] - phi_star[0]) * Lambda
            basis = self.sinusoidal_basis(Lambda)
            phi_star_cur[alpha] += np.dot(basis.T, sigma_ij)

        return phi_star_cur

    def elastic_smoothing(self):
        """
        Apply elastic smoothing to the string nodes contained in self.path.
        
        Returns:
        - smoothed_path: The smoothed string nodes.
        """
        # Generate the smoothed curve
        # Apply elastic smoothing to ensure smooth transitions between nodes
        smoothed_path = np.copy(self.path)
        for bin_idx in range(1, self.n_bins - 1):
            smoothed_path[bin_idx] += self.smoothing_factor * (
                self.path[bin_idx + 1] + self.path[bin_idx - 1] - 2 * self.path[bin_idx]
            )
        # Redistribute nodes uniformly along the arc length of the string
        return self.redistribute_nodes_along_arc_length(smoothed_path)
    
    def compute_arc_length(self, points):
        """
        Compute the arc length of the string by summing the distances between consecutive points.
        
        Parameters:
        - points: A numpy array of shape (n_bins, n_dimensions) representing the string nodes.
        
        Returns:
        - arc_lengths: A numpy array of the cumulative arc lengths along the string.
        """
        arc_lengths = np.zeros(len(points))
        for i in range(1, len(points)):
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(points[i] - points[i - 1])
        return arc_lengths

    def redistribute_nodes_along_arc_length(self, smoothed_path):
        """
        Redistributes the string nodes uniformly along the arc length.
        
        Parameters:
        - smoothed_path: The smoothed path as a numpy array of shape (n_bins, n_dimensions).
        
        Returns:
        - redistributed_path: A numpy array of shape (n_bins, n_dimensions) with uniformly spaced nodes along the arc length.
        """
        arc_lengths = self.compute_arc_length(smoothed_path)
        total_length = arc_lengths[-1]
        uniform_distances = np.linspace(0, total_length, self.n_bins)

        # Interpolate the smoothed path as a function of arc length
        tck, _ = splprep(smoothed_path.T, u=arc_lengths, s=0)
        redistributed_path = np.array(splev(uniform_distances, tck)).T

        return redistributed_path

    def update_string(self, walker_positions, walker_weights):
        """
        Updates the string using the weighted average of walker positions in each bin and applies smoothing.
        
        Parameters:
        - walker_positions: An array of walker positions (shape: [n_replicas, n_dimensions]).
        - walker_weights: An array of walker weights (shape: [n_replicas]).
        
        Updates:
        - self.path: The positions of Voronoi seed points (string nodes) are updated and smoothed.
        """
        # Step 1: Compute the weighted average positions for each bin (proposed new node positions)
        n_replicas = walker_positions.shape[0]
        proposed_path = np.copy(self.path)
        for bin_idx in range(1, self.n_bins - 1):
            walkers_in_bin = [i for i in range(n_replicas) if self.bin_assignment[i] == bin_idx]
            
            if walkers_in_bin:
                # Extract positions and weights of walkers in this bin
                bin_positions = np.array([walker_positions[i] for i in walkers_in_bin])
                bin_weights = np.array([walker_weights[i] for i in walkers_in_bin])
                
                # Normalize weights
                total_weight = np.sum(bin_weights)
                if total_weight > 0:
                    normalized_weights = bin_weights / total_weight
                    proposed_path[bin_idx] = np.average(bin_positions, axis=0, weights=normalized_weights)

        # Step 2: Average the new node positions with the history (time-averaging)
        for bin_idx in range(1, self.n_bins - 1):
            if len(self.position_history[bin_idx]) >= self.time_window:
                self.position_history[bin_idx].pop(0)
            self.position_history[bin_idx].append(proposed_path[bin_idx])
            
            # Average with history
            avg_position = np.mean(self.position_history[bin_idx], axis=0)
            self.path[bin_idx] = self.path[bin_idx] + self.update_rate * (avg_position - self.path[bin_idx])

        # Step 3: Apply smoothing to ensure smooth transitions between nodes
        if self.smoothing_type == 'sinusoidal':
            # Sinusoidal smoothing
            self.path = self.smooth_curve_fit()
        elif self.smoothing_type == 'elastic':
            # Elastic smoothing
            self.path = self.elastic_smoothing()
        else:
            raise ValueError(f'Invalid smoothing type: {self.smoothing_type}')

        # Step 4: Update the history after applying smoothing and redistribution
        for bin_idx in range(self.n_bins):
            if len(self.position_history[bin_idx]) >= self.time_window:
                self.position_history[bin_idx].pop(0)
            self.position_history[bin_idx].append(self.path[bin_idx])

    def run_step(self, walker_positions, walker_weights):
        """
        Executes one step of the binning and string update process.
        
        Parameters:
        - walker_positions: An array of walker positions (shape: [n_replicas, n_dimensions]).
        - walker_weights: An array of walker weights (shape: [n_replicas]).
        
        Returns:
        - Updated string (Voronoi seed positions).
        """
        # Step 1: Update bin assignments for each walker
        self.update_bins(walker_positions)
        
        # Step 2: Update the string (Voronoi seed points) based on walker positions and weights
        self.update_string(walker_positions, walker_weights)

        # Step 3: Remap walkers to bins after the string has been updated
        self.update_bins(walker_positions)
        
        return self.path
