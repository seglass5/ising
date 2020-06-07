"""
ising_model.py

Simulation of the 2D Ising model using both the Metropolis-Hastings and Wolff
algorithms. Data generation to investigate autocovariance, thermalisation,
heat capacity, energy, magnetisation, as well as hysteresis properties.
Animations of thermalisation and different temperatures, and comparative
thermalisation of a large grid using both MH and Wolff algorithms
"""


import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import animation

# Create an NxN grid, and randomly assign spins
def create_grid(n):
	spin = np.array([1,-1])
	grid_of_spins = np.random.choice(spin, size=(n,n))
	return grid_of_spins

# Create an NxN homogeneous grid
def create_homo_grid(n):
    return np.ones((n,n),dtype=int)


# Calculate delta E for a spin in grid at [i,j]
# Decorated function for Numba speed increase
@jit(nopython=True)
def calculate_delta_energy(grid, J, H, i, j):
    n = len(grid)
    delta_energy  = 2*J*grid[i, j]*(grid[i, (j - 1) % n ] + grid[i, (j + 1) % n] + grid[(i - 1) % n, j] + grid[(i + 1) % n, j])+2*H*grid[i, j]
    return delta_energy

# Review delta E for each spin and update n^2 spins in one iteration
# Numba used
@jit
def conditional_flip(spins,J, H, temp):
	#Number of rows in delta_energy
    n = len(spins)
    for u in range(n**2):
    #Generate a new random number each time
        i = np.random.randint(n)
        j = np.random.randint(n)
        delta_energy = calculate_delta_energy(spins, J, H, i, j)
        if (delta_energy < 0) or (np.exp((-delta_energy)/temp) > np.random.random()):
            spins[i,j] *= -1


# Calculate magnetisation of a grid, using absolute for mag-temp, and not for
# hysteresis
# Numba used
@jit
def magnetisation(grid, magnetic_moment):
    # Need abs when looking at mag temp
    M = np.abs((magnetic_moment * np.sum(grid)))
    #M = magnetic_moment * np.sum(grid)
    return M

# Calculate the total system energy of a given grid
def total_system_energy(grid, magnetic_moment,field,exchange_energy):
    n = len(grid)
    #Current energy grid same size as spins
    current_energy = np.empty_like(grid)
    # Go through each element in the grid
    for i in range(0,n):
        for j in range(0,n):
            # Calculate a grid of energies corresponding to spins
            current_energy[i,j] = -magnetic_moment*field*grid[i,j]-exchange_energy*grid[i,j]*(grid[i,(j-1)%n] + grid[i,(j+1)%n] + grid[(i-1)%n,j] + grid[(i+1)%n,j])
    #Factor of 1/2 to avoid overcounting
    total_energy = 0.5 * np.sum(current_energy)
    return total_energy


# Define a function to output grid, vector of magnetisation and energy after
# n iterations
# Numba used
@jit
def update_grid_n_times(n, spins, exchange_energy, magnetic_moment, field, temp):
    # Initialise magnetisation and system_energy vectors
    magnetisation_vector = np.empty(n)
    system_energy = np.empty(n)
    i=0
    # Iterate n times
    while i < n :
        conditional_flip(spins,exchange_energy,field,temp)
        system_energy[i] = total_system_energy(spins,magnetic_moment, field,exchange_energy)
        magnetisation_vector[i] = magnetisation(spins, magnetic_moment)
        i +=1
    return spins, magnetisation_vector, system_energy

# Function to calculate the connected component containing a spin at [i,j]
# Numba used
@jit	
def get_cluster(i,j,grid, N):
    # Get old spin
    old_spin = grid[i,j]
    # Put into cluster
    cluster_a = [(i,j)]
    # Put into visited
    visited = [(i,j)]
    # Initialise empty cluster
    cluster_b = []
    # Move from cluster a to cluster b depending on whether visited and spin
    # aligned
    while len(cluster_a) > 0:
        i, j = cluster_a.pop(0)
        cluster_b.append((i,j))
        
	    # Get neighbours
        ip1 = (i+1)%N
        im1 = (i-1)%N
        jp1 = (j+1)%N
        jm1 = (j-1)%N
        neighbors = [(ip1,j),(im1,j),(i,jp1),(i,jm1)]
        
        # Loop through neighbours
        for m, n in neighbors:
            # Check if spin aligned and not visited, add to clusrer b if so 
            if (grid[m,n] == old_spin) and ((m,n) not in visited):
                visited.append((m,n))
                cluster_a.append((m,n))
    return cluster_b

# Calculate sizes of clusters
def domain_sizer(grid):
    grid_list = []
    length_list = []
    # Go through all points in grid, add to a list
    for i in range(len(grid)):
        for j in range(len(grid)):
            grid_list.append((i,j))
    # Get clusters of all points in grid, remove from list once checked
    while len(grid_list) > 0:
        cluster = get_cluster(grid_list[0][0],grid_list[0][1],grid, len(grid))
        length_list.append(len(cluster))
        grid_list = [x for x in grid_list if x not in cluster]
    # Return list of cluster sizes
    return length_list

# Calculate maximum and average domain size at discrete temperatures within
# a range, specify range with temp_low and temp_high, initial_grid input
# Numba used
@jit
def domain_size_temperature(temp_low,temp_high, initial_grid):
    # Define temperature range
    T = np.linspace(temp_low,temp_high,200)
    # Initialise empty vectors
    max_domain_size = np.empty(len(T))
    average_domain_size = np.empty(len(T))
    # Go through temperature range, add max domain size and average domain size
    # to a vector
    for i in range(len(T)):
        equilibrium_grid = update_grid_n_times(150,initial_grid,1,1,0,T[i])[0]
        max_domain_size[i] = max(domain_sizer(equilibrium_grid))
        average_domain_size[i] = np.mean(domain_sizer(equilibrium_grid))
    # Return temperature vector and domain size vectors
    return T,max_domain_size,average_domain_size

# Calculate the heat capacity over a discrete range of temperatures
# Specify temperature range with temp_high. temp_low, temp_interval
# Average over a number of iterations
# Numba used    
@jit
def heat_capacity(temp_low, temp_high, grid, temp_interval, iterations):
    # Specify temperature range
    T = np.linspace(temp_low, temp_high, temp_interval)
    # Initialise empty vectors
    N = len(grid)
    heat_cap = np.empty(len(T))
    mean_system_energy = np.empty((len(T)))
    for i in range(len(T)):
        # Calculate heat capacity and mean system energy at each T
        primed_grid = update_grid_n_times(150, grid, 1, 1, 0, T[i])[0]
        energy_vector = update_grid_n_times(iterations, primed_grid, 1, 1, 0, T[i])[2]
        # Add to vectors
        mean_system_energy[i] = (np.mean(energy_vector))/(N**2) 
        var_energy = np.var(energy_vector, dtype=np.float64)
        heat_cap[i] = (var_energy)/((N*T[i])**2)
    # Return heat capacity, system energy and temperature
    return heat_cap,mean_system_energy, T

# Thermalise a grid and then output a vector of |M| and T
# Specify temperature range with temp_high. temp_low
# Average over n iterations
# Numba used
@jit
def magnetisation_temperature(temp_low, temp_high, grid, n):
    # Specify temp range
    T = np.linspace(temp_low, temp_high, 200)
    N = len(grid)
    # Initialise vectors
    mag_vect = np.empty(len(T))
    susceptibility = np.empty(len(T)) 
    for i in range(len(T)):
        # Thermalise for each T
        primed_grid = update_grid_n_times(150, grid, 1,1,0,T[i])[0]
        # Calculate magnetisation vector
        magnetisation = update_grid_n_times(n, primed_grid, 1,1,0,T[i])[1]
        # Append magnetisation and magnetic susceptibility values
        mag_vect[i] = (np.mean(magnetisation))/(N**2)
        var_mag = np.var(magnetisation, dtype=np.float64)
        susceptibility[i] = var_mag/(T[i]*(N**2))
    return mag_vect, T, susceptibility


# Step through H discrete range of H values specified by H_range
# Numba used
@jit
def magnetisation_in_field_range(temperature, H_range, N, initial_spins, number):
    # Sinusoidal applied field
	H = np.cos(np.linspace(0,H_range,number))
	mean_magnetisation = np.empty(len(H))
	# For each H, run update_grid_n_times, calculate the mean magnetisation
    # vector, divide by N^2
	for i in range(len(H)):
		magnet_vector = update_grid_n_times(N,initial_spins,1,1,H[i],temperature)[1]
		mean_magnetisation[i] = np.mean(magnet_vector)/((len(initial_spins))**2)
	return H, mean_magnetisation

# Define single iteration of Wolff algorithm on NxN grid, at specified J and T
def WolffMove(grid, N, J, T):
    # Pick a random spin and remember its direction as old spin 
    i = np.random.randint(0,N)
    j = np.random.randint(0,N)
    oldSpin = grid[i,j]
    # Push onto list of spins to flip
    toFlip = [(i,j)]
    # Set spins flipped to 0
    spinsFlipped = 0
    # While there are still spins to flip
    while len(toFlip) > 0:
        # Remove the first spin
        i, j = toFlip.pop(0)
	    # Check if flipped in between, if not flip it
        if grid[i,j] == oldSpin:
            grid[i,j] *= -1
            spinsFlipped += 1
            # Get neighbours
            ip1 = (i+1)%N
            im1 = (i-1)%N
            jp1 = (j+1)%N
            jm1 = (j-1)%N
            neighbors = [(ip1,j),(im1,j),(i,jp1),(i,jm1)]
            # For each of its neighbours, if spin aligned put on stack with
            # probability p
            for m, n in neighbors:
                if grid[m,n] == oldSpin:
                   if np.random.random() < (1-np.exp(-2*J/T)):
                        toFlip.append((m,n))
    return spinsFlipped

# Calculate the magnetisation after each of n Wolff iterations
def wolff_magnetisation(grid, N, J, T, n):
    i = 0
    # Initialise vector
    mag_vect = np.empty(n)
    # Iterate n times
    while i < n:
        WolffMove(grid, N, J, T)
        # Calculate |M|, append to vector on each iteration
        mag_vect[i] = np.abs((1/(N**2))*np.sum(grid))
        i+=1
    return mag_vect

def main():
    # Define variables
    J = 1
    magnetic_moment = 1
    H = 0
    T_crit = 2.27
    # Define values of N used
    N = np.array([2,4,8,16,32,64,128])
    # Define array of random initial grids
    initial_grids = np.array([create_grid(2),create_grid(4),create_grid(8),create_grid(16),create_grid(32),create_grid(64),create_grid(128)])
    # Define array of homogeneous initial grids
    initial_homo_grids = np.array([create_homo_grid(2),create_homo_grid(4),create_homo_grid(8),create_homo_grid(16),create_homo_grid(32),create_homo_grid(64),create_homo_grid(128)])
        
    """ Generate autocovariance data """
    
    """ 1000 iterations for 32 x 32 random grid at T = 2.27"""
    magnetisation_vector_2 = update_grid_n_times(1000,initial_grids[0], J, magnetic_moment,H, 2.27)[1]
    magnetisation_vector_4 = update_grid_n_times(1000,initial_grids[1], J, magnetic_moment,H, 2.27)[1]
    magnetisation_vector_8 = update_grid_n_times(1000,initial_grids[2], J, magnetic_moment,H, 2.27)[1]
    magnetisation_vector_16 = update_grid_n_times(1000,initial_grids[3], J, magnetic_moment,H, 2.27)[1]
    magnetisation_vector_32 = update_grid_n_times(1000,initial_grids[4], J, magnetic_moment,H, 2.27)[1]
    magnetisation_vector_64 = update_grid_n_times(1000,initial_grids[5], J, magnetic_moment,H, 2.27)[1]
    magnetisation_vector_128 = update_grid_n_times(1000,initial_grids[6], J, magnetic_moment,H, 2.27)[1]
    n = np.linspace(1,1000,1000)
    M_against_steps = np.array([n, magnetisation_vector_2, magnetisation_vector_4,magnetisation_vector_8, magnetisation_vector_16,magnetisation_vector_32,magnetisation_vector_64,magnetisation_vector_128])
    np.save('MnTcdata.npy', M_against_steps)
    
    """ 200 iterations for 32 x 32 random grid at T = 0.5,1.5,2,2.5,3,4 """
    magnetisation_temp_0 = update_grid_n_times(200,initial_grids[4], J, magnetic_moment,H, 0.5)[1]
    magnetisation_temp_1 = update_grid_n_times(200,initial_grids[4], J, magnetic_moment,H, 1.5)[1]
    magnetisation_temp_2 = update_grid_n_times(200,initial_grids[4], J, magnetic_moment,H, 2)[1]
    magnetisation_temp_3 = update_grid_n_times(200,initial_grids[4], J, magnetic_moment,H, 2.5)[1]
    magnetisation_temp_4 = update_grid_n_times(200,initial_grids[4], J, magnetic_moment,H, 3)[1]
    magnetisation_temp_5 = update_grid_n_times(200,initial_grids[4], J, magnetic_moment,H, 4)[1]
    n = np.linspace(1,200,200)
    M_against_steps_T = np.array([n, magnetisation_temp_0, magnetisation_temp_1, magnetisation_temp_2, magnetisation_temp_3,magnetisation_temp_4, magnetisation_temp_5])
    np.save('M_against_steps_T.npy', M_against_steps_T)
    
    """ 200 iterations for 32 x 32 homogeneous grid at range of T """
    magnetisation_homo_0 = update_grid_n_times(200,initial_homo_grids[4], J, magnetic_moment,H, 0.5)[1]
    magnetisation_homo_1 = update_grid_n_times(200,initial_homo_grids[4], J, magnetic_moment,H, 1.5)[1]
    magnetisation_homo_2 = update_grid_n_times(200,initial_homo_grids[4], J, magnetic_moment,H, 2)[1]
    magnetisation_homo_3 = update_grid_n_times(200,initial_homo_grids[4], J, magnetic_moment,H, 2.5)[1]
    magnetisation_homo_4 = update_grid_n_times(200,initial_homo_grids[4], J, magnetic_moment,H, 3)[1]
    magnetisation_homo_5 = update_grid_n_times(200,initial_homo_grids[4], J, magnetic_moment,H, 4)[1]
    n = np.linspace(1,200,200)
    M_against_steps_homo = np.array([n, magnetisation_homo_0, magnetisation_homo_1, magnetisation_homo_2, magnetisation_homo_3, magnetisation_homo_4, magnetisation_homo_5])
    np.save('M_against_steps_homo.npy', M_against_steps_homo)
    
    """ Thermalised Grids """
    grid_a = update_grid_n_times(150, initial_grids[5], J, magnetic_moment, H, 0.5)[0]
    np.save('thermal_grid_a.npy', grid_a)
    grid_b = update_grid_n_times(150, initial_grids[5], J, magnetic_moment, H, 1.5)[0]
    np.save('thermal_grid_b.npy',grid_b)
    grid_c = update_grid_n_times(150, initial_grids[5], J, magnetic_moment, H, 2.0)[0]
    np.save('thermal_grid_c.npy',grid_c)
    grid_d = update_grid_n_times(150, initial_grids[5], J, magnetic_moment, H, 2.27)[0]
    np.save('thermal_grid_d.npy',grid_d)
    grid_e = update_grid_n_times(150, initial_grids[5], J, magnetic_moment, H, 2.5)[0]
    np.save('thermal_grid_e.npy',grid_e)
    grid_f = update_grid_n_times(150, initial_grids[5], J, magnetic_moment, H, 5.0)[0]
    np.save('thermal_grid_f.npy',grid_f)
    

    """Coarse Heat Capacity, averaging over 3000 iterations"""
    heat_cap_2 = heat_capacity(0.5,4.5,initial_grids[0],3000)
    heat_cap_4 = heat_capacity(0.5,4.5,initial_grids[1],3000)
    heat_cap_8 = heat_capacity(0.5,4.5,initial_grids[2],3000)
    heat_cap_16 = heat_capacity(0.5,4.5,initial_grids[3],3000)
    heat_cap_32 = heat_capacity(0.5,4.5,initial_grids[4],3000)
    heat_cap_64 = heat_capacity(0.5,4.5,initial_grids[5],3000)
    #heat_cap_128 = heat_capacity(0.5,4.5,initial_grids[6],5000)
    
    energy_temp = np.array([heat_cap_2[2],heat_cap_2[1],heat_cap_4[1],heat_cap_8[1],heat_cap_16[1],heat_cap_32[1],heat_cap_64[1]])
    Cv_temp = np.array([heat_cap_2[2],heat_cap_2[0],heat_cap_4[0],heat_cap_8[0],heat_cap_16[0],heat_cap_32[0],heat_cap_64[0]])
    np.save('EnergyTemp.npy',energy_temp)
    np.save('CvTemp.npy',Cv_temp)
    
    
    """Fine Heat Capacity, averaging over 5000 iterations"""
    heat_cap_2_fine = heat_capacity(2.15,2.55,initial_grids[0],100,5000)
    heat_cap_4_fine = heat_capacity(2.15,2.55,initial_grids[1],100,5000)
    heat_cap_8_fine = heat_capacity(2.15,2.55,initial_grids[2],100,5000)
    heat_cap_16_fine = heat_capacity(2.15,2.55,initial_grids[3],100,5000)
    heat_cap_32_fine = heat_capacity(2.15,2.55,initial_grids[4],100,5000)
    heat_cap_64_fine = heat_capacity(2.15,2.55,initial_grids[5],100,5000)
    #heat_cap_128_fine = heat_capacity(2,2.5,initial_grids[6],100,5000)
    
    Cv_temp_fine = np.array([heat_cap_2_fine[2],heat_cap_2_fine[0],heat_cap_4_fine[0],heat_cap_8_fine[0],heat_cap_16_fine[0], heat_cap_32_fine[0], heat_cap_64_fine[0]])
    np.save('CvTempFine.npy',Cv_temp_fine)
    
    """Magnetisation against temperature data, averaging over 3000 iteration"""
    mag_temp_2 = magnetisation_temperature(0.5,4.5,initial_grids[0],3000)
    mag_temp_4 = magnetisation_temperature(0.5,4.5,initial_grids[1],3000)
    mag_temp_8 = magnetisation_temperature(0.5,4.5,initial_grids[2],3000)
    mag_temp_16 = magnetisation_temperature(0.5,4.5,initial_grids[3],3000)
    mag_temp_32 = magnetisation_temperature(0.5,4.5,initial_grids[4],3000)
    mag_temp_64 = magnetisation_temperature(0.5,4.5,initial_grids[5],3000)
    
    mag_temp = np.array([mag_temp_2[1], mag_temp_2[0], mag_temp_4[0],mag_temp_8[0], mag_temp_16[0],mag_temp_32[0],mag_temp_64[0]])
    np.save('MagTemp.npy', mag_temp)
    
    """Generate Hysteresis Data"""
    thermalised_grid = update_grid_n_times(150,initial_grids[5],1,1,0,0.5)[0]
        
    hyst_a = magnetisation_in_field_range(0.5,4*np.pi,150,thermalised_grid,500)
    hyst_b = magnetisation_in_field_range(1.5,4*np.pi,150,thermalised_grid,500)
    hyst_c = magnetisation_in_field_range(2.0,4*np.pi,150,thermalised_grid,500)
    hyst_d = magnetisation_in_field_range(2.27,4*np.pi,150,thermalised_grid,500)
    hyst_e = magnetisation_in_field_range(2.5,4*np.pi,150,thermalised_grid,500)
    hyst_f = magnetisation_in_field_range(5.0,4*np.pi,150,thermalised_grid,500)
    
    hyst_mag = np.array([hyst_a[0],hyst_a[1],hyst_b[1],hyst_c[1],hyst_d[1],hyst_e[1],hyst_f[1]])
    np.save('HystMag.npy',hyst_mag)
    
    """Wolff Algorithm"""
    
    wolff_mag_128 = wolff_magnetisation(initial_homo_grids[6],N[6],J,T,1000)
    MH_mag_128 = update_grid_n_times(1000,initial_homo_grids[6],J,magnetic_moment,H,T)[1]
    
    wolff_speed = np.array([time_delay,wolff_mag_128,MH_mag_128])
    np.save('wolffspeed.npy',wolff_speed)
    
    """Grids for Domain Sizes"""
    dom_size_32 = domain_size_temperature(1.5,3.5,initial_grids[4])
    dom_size_64 = domain_size_temperature(1.5,3.5,initial_grids[5])
    dom_size_128 = domain_size_temperature(1.5,3.5,initial_grids[6])
    
    dom_temp_max = np.array([dom_size_32[0],dom_size_32[1]/N[4],dom_size_64[1]/N[5],dom_size_128[1]/N[6]])
    np.save('dom_temp_max.npy',dom_temp_max)
    
    dom_temp_average = np.array([dom_size_32[0],dom_size_32[2]/N[4],dom_size_64[2]/N[5],dom_size_128[2]/N[6]])
    np.save('dom_temp_average.npy',dom_temp_average)
    
    """Create Animations, adjust parameters for different animations"""
        
    def update(i):
        # Change between WolffMove and conditional_flip using comments
        WolffMove(initial_grids[6],128,1,2.27)
        #conditional_flip(initial_grids[6],1,0,2.27)
        matrix.set_array(initial_grids[6])
        
    fig, ax = plt.subplots()
    matrix = ax.matshow(initial_grids[6])

    anim = animation.FuncAnimation(fig, update, frames=150, interval=1)
    anim.save('Wolff.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    
    
main()
