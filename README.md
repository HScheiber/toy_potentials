# Toy Potentials 

This is a library for generating 2D toy potentials, and for generating ensembles of stochastic trajectories over those potentials. We use `deeptime` to run simulations of walkers over the potentials at specified temperatures.

## Available Potentials
 - Muller Brown
 - Symmetric Four Well
 - Asymmetric Four Well
 - `L` potential made from Gaussians 
 - `Maze` potential made from Gaussians (meant to be a challenging test case)
 
 You can also build your own custom potentials from adding up Gaussians. See how this is done in the [potentials.py](./src/toy_potentials/potentials.py) code.

## Requirements
**Base Requirements:**
- python >= 3.5.0
- numpy
- scipy 
- deeptime (for generating trajectories) 
- tqdm (for loading bars)
- joblib (for parallelization)

**For visualizing examples:**
- matplotlib


## Installation
To install this software, clone the repository then use pip.
```bash
cd /path/to/toy_potentials
python -m pip install .
```

## Getting Started

For example usage of this library, see the [example jupyter notebook.](./Examples/sim_visualization.ipynb) Note that matplotlib library is required for visualization. By changing up the line
```python
param_file = './fpvt_muller_brown_potential.json'
```
you can select from 5 different pre-prepared examples. Below are the different parameters you can use to modify the simulations.
### Primary Parameters
- `experiment_name` This is the simulation name.
- `n_walkers`:  How many walkers to initialize. Each walker will have its own trajectory.
- `n_steps_per_iter`: How many time steps to run before resampling walkers with the weighted ensemble resampling method. If not using resampling, you can set this to the total number of time steps you want to run and set `n_iters` to 1.
- `n_iters`:  How many total iterations to run. (Total time steps will be `n_steps_per_iter`*`n_iters`)
- `kT`: Temperature, increasing this will give the walkers larger random kicks.
- `gamma`: Langevin friction parameter. Leave this as 1 as it basically does the same as kT.
- `integration_step`: The time step to use for integrating the equation of motion.
- `min_weight`: When using weighted ensemble resampling, this sets the minimum walker weight before walker splitting is no longer allowed. Walkers will never have less than this weight.
- `potential_type`: Select your potential here. Potentials are defined in [potentials.py](./src/toy_potentials/potentials.py). Default options are `four_well`, `four_well_asym`, `muller_brown`, `maze`, `L`.
- `binning_strategy`: If using weighted ensemble resampling, this sets your binning strategy. Each binning strategy has its own associated parameters. Available options are `fpvt` (furthest point voronoi tessellation), `rectilinear` (simple MxN grid of equally spaced bins), `string` (a sophisticated [string-based method](https://stringmethodexamples.readthedocs.io/en/latest/)).
- `n_walkers_per_bin`: Number of walkers per bins if using weighted ensemble resampling. Total walkers will then be `n_bins`*`n_bins`
- `use_resampling`: Setting this to true turns on Weighted Ensemble resampling. Turn it to False to run `n_walkers` number of independent trajectories.
- `steady_state`: This turns on steady-state mode, wherein walkers that reach a predetermined target region (the sink region) are recycled back to the source region of space. Useful for computing rate connstants.
- `n_processes`: Number of parallel processors to use.

### Bin-specific parameters

__FPVT Binning__
- `n_bins`: Number of bins to use, if using FPVT weighted ensemble resampling.

__Rectilinear Binning__
- `x_bounds`: The upper and lower bounds for your rectilinear bins in the x-dimension (e.g. [0,1]).
- `y_bounds`: The upper and lower bounds for your rectilinear bins in the y-dimension (e.g. [0,1]).
- `n_bins_x`: The number of bins to use along the x-dimension.
- `n_bins_y`: The number of bins to use along the y-dimension. Total number of bins is - `n_bins_x`*`n_bins_y`.

__string_binning__
- `start_state`: the [X,Y] coordinates of the first bead in the string.
- `end_state`: the [X,Y] coordinates of the last bead in the string.
- `n_bins`: Number of string bins to use, including the first and last bins.
- `update_rate`: A parameter that tells the algorithm how rapidly to update the string beads. Results are very sensitive to this parameter.
- `time_window`: Number of previous iterations to consider for time averaging of string bead positions.
- `smoothing_type`: Type of smoothing to apply to the string ('elastic' or 'sinusoidal'). Elastic is much faster to compute.
- `smoothing_factor` = Controls how aggressively the string smoothing is applied (κ in the formula) if using elastic smoothing.
- `num_basis_functions` = Number of sinusoidal basis functions to use for curve fitting if using sinusoidal smoothing.

## Authors

- Hayden Scheiber: hscheibe@amgen.com
