import numpy as np

def gaussian(x, y, x0, y0, A, sigma):
    """Compute the value of a 2D Gaussian function."""
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def gaussian_force(x, y, x0, y0, A, sigma):
    """Compute the force (negative gradient) of a 2D Gaussian function."""
    factor = A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    dV_dx = factor * (x - x0) / (sigma**2)
    dV_dy = factor * (y - y0) / (sigma**2)
    return np.array([dV_dx, dV_dy])

def four_well_asym_potential(xy, alpha_x=(1/16), alpha_y=(1/8), beta_x=2, beta_y=2, gamma_x=(3/16), gamma_y=(3/8)):
    """Compute the four-well potential for coordinates (x, y)."""
    return (xy[:,0]**4) - alpha_x*(xy[:,0]**3) - beta_x*(xy[:,0]**2) + gamma_x*xy[:,0] + \
           (xy[:,1]**4) - alpha_y*(xy[:,1]**3) - beta_y*(xy[:,1]**2) + gamma_y*xy[:,1]

def four_well_asym_forces(xy, alpha_x=(1/16), alpha_y=(1/8), beta_x=2, beta_y=2, gamma_x=(3/16), gamma_y=(3/8)):
    """Compute the gradient (force) of the four-well potential at (x, y)."""
    dV_dx = -4*(xy[0]**3) + 3*alpha_x*(xy[0]**2) + 2*beta_x*xy[0] - gamma_x
    dV_dy = -4*(xy[1]**3) + 3*alpha_y*(xy[1]**2) + 2*beta_y*xy[1] - gamma_y
    return np.array([dV_dx, dV_dy])

def four_well_potential(xy):
    """Compute the four-well potential for coordinates (x, y)."""
    return (xy[:,0]**2 - 1)**2 + (xy[:,1]**2 - 1)**2

def four_well_forces(xy):
    """Compute the gradient (force) of the four-well potential at (x, y)."""
    dV_dx = -4*(xy[0]**3) + 4*xy[0]
    dV_dy = -4*(xy[1]**3) + 4*xy[1]
    return np.array([dV_dx, dV_dy])

def maze_potential(xy, walls, A, sigma):
    """Compute the maze-like potential for coordinates (x, y) based on wall positions."""
    x, y = xy[:, 0], xy[:, 1]
    V = np.zeros_like(x)
    for wall, a, sig in zip(walls, A, sigma):
        x0, y0 = wall
        V += gaussian(x, y, x0, y0, a, sig)
    return V

def maze_forces(xy, walls, A, sigma):
    """Compute the force of the maze-like potential at (x, y) based on wall positions."""
    x, y = xy[0], xy[1]
    dV_dx = 0
    dV_dy = 0
    for wall, a, sig in zip(walls, A, sigma):
        x0, y0 = wall
        grad = gaussian_force(x, y, x0, y0, a, sig)
        dV_dx += grad[0]
        dV_dy += grad[1]
    return np.array([dV_dx, dV_dy])

def muller_brown_potential(xy, 
                           A=np.array([-200, -100, -170, 15]), 
                           a=np.array([-1, -1, -6.5, 0.7]), 
                           b=np.array([0, 0, 11, 0.6]), 
                           c=np.array([-10, -10, -6.5, 0.7]), 
                           x0=np.array([1, 0, -0.5, -1]), 
                           y0=np.array([0, 0.5, 1.5, 1]), 
                           h=0.04):
    """
    Compute the Müller-Brown potential for coordinates (x, y).
    """
    x, y = xy[:, 0], xy[:, 1]
    potential = 0
    for i in range(4):
        dx = x - x0[i]
        dy = y - y0[i]
        exponent = a[i] * dx**2 + b[i] * dx * dy + c[i] * dy**2
        potential += A[i] * np.exp(exponent)
    return h * potential

def muller_brown_forces(xy, 
                        A=np.array([-200, -100, -170, 15]), 
                        a=np.array([-1, -1, -6.5, 0.7]), 
                        b=np.array([0, 0, 11, 0.6]), 
                        c=np.array([-10, -10, -6.5, 0.7]), 
                        x0=np.array([1, 0, -0.5, -1]), 
                        y0=np.array([0, 0.5, 1.5, 1]), 
                        h=0.04):
    """
    Compute the gradient (force) of the Müller-Brown potential at (x, y).
    """
    x, y = xy[0], xy[1]
    dV_dx, dV_dy = 0, 0
    for i in range(4):
        dx = x - x0[i]
        dy = y - y0[i]
        exponent = a[i] * dx**2 + b[i] * dx * dy + c[i] * dy**2
        exp_term = np.exp(exponent)
        dV_dx += A[i] * exp_term * (2 * a[i] * dx + b[i] * dy)
        dV_dy += A[i] * exp_term * (b[i] * dx + 2 * c[i] * dy)
    
    return -h * np.array([dV_dx, dV_dy])

def find_local_minima_2d(data):
    """Find local minima in a 2D array."""
    local_min = np.ones_like(data, dtype=bool)
    for shift_x in [-1, 0, 1]:
        for shift_y in [-1, 0, 1]:
            if shift_x == 0 and shift_y == 0:
                continue
            shifted = np.roll(np.roll(data, shift_x, axis=1), shift_y, axis=0)
            local_min &= data < shifted
    return local_min

def load_potential(potential_name):
    """Load the potential function and its gradient based on the given potential name."""
    if potential_name == 'four_well':
        potential = four_well_potential
        forces = four_well_forces

        E_max = 15
        x_range = (-2.8, 2.8)
        y_range = (-2.5, 2.5)
        basis_location = [-1, -1]
        basis_scale = [0.1, 0.1]
        tstate_location = [1.0, 1.0]
        saddle_points = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])

        def reached_tstate(coords, x_center=1, y_center=1, a=0.05, b=0.05):
            """
            This function takes a (M,N) array of N-dimensional coordinates as input and a boolean mask
            over the coordinates that returns True if the coordinate is in the target state.
            Here the target state is the basin centered at (1.72, 0.95).
            """
            return ((coords[:,0] - x_center)**2 / a**2) + ((coords[:,1] - y_center)**2 / b**2) < 1
        tstate = reached_tstate

    elif potential_name == 'four_well_asym':

        potential = lambda xy: four_well_asym_potential(xy, 
                alpha_x=(1/100), 
                alpha_y=(1/32), 
                beta_x=6, 
                beta_y=2, 
                gamma_x=(3/16), 
                gamma_y=(3/8)
            )

        forces = lambda xy: four_well_asym_forces(xy, 
            alpha_x=(1/100), 
            alpha_y=(1/32), 
            beta_x=6, 
            beta_y=2, 
            gamma_x=(3/16), 
            gamma_y=(3/8)
        )

        E_max = 15
        x_range = (-2.8, 2.8)
        y_range = (-2.5, 2.5)
        basis_location = [-1.75, -1]
        basis_scale = [0.1, 0.1]
        tstate_location = [1.72, 0.125]
        saddle_points = np.array([[-1.75, 0.09], [0, 1], [0, -1], [1.72, 0.09]])

        def reached_tstate(coords, x_center=1.72, y_center=0.95, a=0.05, b=0.05):
            """
            This function takes a (M,N) array of N-dimensional coordinates as input and a boolean mask
            over the coordinates that returns True if the coordinate is in the target state.
            Here the target state is the basin centered at (1.72, 0.95).
            """
            return ((coords[:,0] - x_center)**2 / a**2) + ((coords[:,1] - y_center)**2 / b**2) < 1
        tstate = reached_tstate

    elif potential_name == 'maze':

        walls = [(i, 0) for i in range(0, 11)] +\
                [(0, i) for i in range(1, 11)] +\
                [(i, 10) for i in range(1, 11)] +\
                [(10, i) for i in range(1, 10)] +\
                [(2, i) for i in range(2, 9)] +\
                [(i, 2) for i in range(3, 9)] +\
                [(4, i) for i in range(4, 10)] +\
                [(i, 4) for i in range(6, 10)] +\
                [(i, 7) for i in range(5, 7)] +\
                [(i, 6) for i in range(8, 10)] + \
                [ (8,8), (7,9), (7,6), (3,3.5), (8.5,3), (5,4), (1,5)]

        # define the heights and sigmas
        heights = np.ones(len(walls))*12
        gsigmas = np.ones(len(walls))*0.05
        heights[-5] = 8
        gsigmas[-5] = 0.06
        heights[-4] = 5
        gsigmas[-4] = 0.05
        heights[-3] = 5
        gsigmas[-3] = 0.05
        heights[-2] = 3
        gsigmas[-2] = 0.05
        heights[-1] = 5
        gsigmas[-1] = 0.05

        # Modifications
        # sel_mask = (np.array(walls) == np.array([(4,9)])).all(axis=1)
        # heights[sel_mask] = 5
        # gsigmas[sel_mask] = 0.05

        # Rescale to range (0, 1)
        walls = [(i/10,j/10) for (i,j) in walls]

        # Create the walls
        potential = lambda xy: maze_potential(xy, walls, A=heights, sigma=gsigmas)
        forces = lambda xy: maze_forces(xy, walls, A=heights, sigma=gsigmas)

        E_max = 18
        x_range = (0, 1)
        y_range = (0, 1)
        basis_location = [0.55, 0.85]
        basis_scale = [0.015, 0.015]
        tstate_location = [0.125, 0.125]
        saddle_points = np.array([[0.75, 0.85], [0.66, 0.64], [0.1, 0.5], [0.505, 0.4], [0.31, 0.37], [0.835, 0.29]])

        def reached_tstate(coords, x_center=0.125, y_center=0.125, a=0.05, b=0.05):
            """
            This function takes a (M,N) array of N-dimensional coordinates as input and a boolean mask
            over the coordinates that returns True if the coordinate is in the target state.
            Here the target state is the basin centered at (0.623, 0.028).
            """
            return ((coords[:,0] - x_center)**2 / a**2) + ((coords[:,1] - y_center)**2 / b**2) < 1
        tstate = reached_tstate

    elif potential_name == 'L':
        walls = np.zeros((11,11), dtype=bool)
        walls[:,     0] = True
        walls[0:8,   3] = True
        walls[7:10, 10] = True
        walls[0,   1:3] = True
        walls[7,    3:] = True
        walls[10,    :] = True
        walls[3,     1] = True
        walls[4,     1] = True
        walls[4,     2] = True
        walls[6,     1] = True
        walls[7,     2] = True
        walls[8,     2] = True
        walls[9,     3] = True
        walls[8,     5] = True
        walls[9,     5] = True
        walls = np.argwhere(walls)

        heights = np.ones((11,11))*40
        heights[3,1] = 8
        heights[4,1] = 8
        heights[4,2] = 8
        heights[6,1] = 10
        heights[7,2] = 8
        heights[8,2] = 10
        heights[9,3] = 8
        heights[8,5] = 8
        heights[9,5] = 8
        heights= np.array([heights[idx[0],idx[1]] for idx in walls])

        gsigma = np.ones((11,11))*0.04
        gsigma[3,1]= 0.04
        gsigma[4,1]= 0.04
        gsigma[4,2]= 0.04
        gsigma[6,1]= 0.05
        gsigma[7,2]= 0.05
        gsigma[8,2]= 0.06
        gsigma[9,3]= 0.07
        gsigma[8,5]= 0.05
        gsigma[9,5]= 0.05
        gsigma = np.array([gsigma[idx[0],idx[1]] for idx in walls])

        walls = [(i/10,j/10) for (i,j) in walls]

        potential = lambda xy: maze_potential(xy, walls, A=heights, sigma=gsigma)
        forces = lambda xy: maze_forces(xy, walls, A=heights, sigma=gsigma)

        x_range = (-0.1, 1.1)
        y_range = (-0.1, 1.1)
        basis_location = [0.2, 0.15]
        basis_scale = [0.015, 0.015]
        E_max = 20
        tstate_location = [0.85, 0.85]
        saddle_points = np.array([[0.4, 0.15], [0.65, 0.15],[0.85, 0.25],[0.85,0.5]])

        def reached_tstate(coords, x_center=0.85, y_center=0.85, a=0.05, b=0.05):
            """
            This function takes a (M,N) array of N-dimensional coordinates as input and a boolean mask
            over the coordinates that returns True if the coordinate is in the target state.
            Here the target state is the basin centered at (0.85, 0.85).
            """
            return ((coords[:,0] - x_center)**2 / a**2) + ((coords[:,1] - y_center)**2 / b**2) < 1
        tstate = reached_tstate

    elif potential_name == 'muller_brown':
        potential = muller_brown_potential
        forces = muller_brown_forces

        x_range = (-2.7, 1.3)
        y_range = (-0.7, 2.7)
        basis_location = [-0.558, 1.441]
        basis_scale = [0.02, 0.02]
        tstate_location = [0.623, 0.028]
        E_max = 10
        saddle_points = np.array([[-0.822, 0.624], [0.212, 0.293]])

        def reached_tstate(coords, x_center=0.623, y_center=0.028, a=0.2, b=0.1):
            """
            This function takes a (M,N) array of N-dimensional coordinates as input and a boolean mask
            over the coordinates that returns True if the coordinate is in the target state.
            Here the target state is the basin centered at (0.623, 0.028).
            """
            return ((coords[:,0] - x_center)**2 / a**2) + ((coords[:,1] - y_center)**2 / b**2) < 1
        tstate = reached_tstate
    else:
        raise ValueError(f'Potential {potential_name} not recognized.')
    
    return potential, forces, tstate, tstate_location, x_range, y_range, basis_location, basis_scale, E_max, saddle_points