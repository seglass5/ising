import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

#Define a function to create an NxN grid, and randomly assign spins
def create_grid(n):
	spin = np.array([1,-1])
	grid_of_spins = np.random.choice(spin, size=(n,n))
	return grid_of_spins

#Create grids
thirty = create_grid(30)
three = create_grid(3)
five = create_grid(5)
ten = create_grid(10)
#Plot
#Put this into a function
#ax1 = plt.imshow(five)
#plt.ylim(-0.5,29.5)
#plt.savefig('five.png')

#Define a function to calculate the Delta E of a spin given the grid and its location
#Matriculate this

def calculate_delta_energy(grid, exchange_energy, magnetic_moment, field):
	n = len(grid)
	current_energy = np.empty_like(grid)
	delta_energy = np.empty_like(grid)
	for i in range(0,n):
		for j in range(0,n):
			current_energy[i,j] = -magnetic_moment*field*grid[i,j] -(1/2)*exchange_energy*(grid[i,j]*grid[i,(j-1)%n] + grid[i,j]*grid[i,(j+1)%n] + grid[i,j]*grid[(i-1)%n,j] + grid[i,j]*grid[(i+1)%n,j])
	delta_energy = - 2*current_energy
	return delta_energy, current_energy
	
#delta, bigE = calculate_delta_energy(nine, 1, 1, 0)
#print(delta)
#print(bigE)

#Define a function to ask if the energy change at a given position is
#Return the flip_matrix
def energy_evaluator(energy_change_grid, temp):
	n = len(energy_change_grid)
	flip_matrix = np.ones_like(energy_change_grid)
	for i in range(0,n):
		for j in range(0,n):
			p = np.random.uniform()
			if (energy_change_grid[i,j]<0):
				flip_matrix[i,j] = -1
			elif (np.exp(-energy_change_grid[i,j]/(temp*(sc.k)))>p):
				flip_matrix[i,j] = -1
	return flip_matrix

#print(energy_evaluator(delta, 1))
#One time loop is calculate energy matrix, calculate flip matrix and multiply
#Define a function that takes the required number of time loops and the initial matrix and exchange energy magnetic moment, field and temperature
#And runs it for the specified number of loops, preferably updating plot as it goes
#Define a function to calculate magnetisation

def magnetisation(grid, magnetic_moment):
	M = magnetic_moment * np.sum(grid)
	return M
	
def looper(n, initial_spins, exchange_energy, magnetic_moment, field, temp):
	deltaE, E = calculate_delta_energy(initial_spins, exchange_energy, magnetic_moment, field)
	flip_matrix = energy_evaluator(deltaE,temp)
	spins = np.multiply(initial_spins, flip_matrix)
	magnetisation_vector = np.empty(n)
	#fig,ax = plt.subplots()
	i=0
	while i < n :
		deltaE, E = calculate_delta_energy(spins, exchange_energy, magnetic_moment, field)
		flip_matrix = energy_evaluator(deltaE,temp)
		spins = np.multiply(spins, flip_matrix)
		magnetisation_vector[i] = magnetisation(spins, magnetic_moment)
		#ax.cla()
		#ax.imshow(spins)
		#plt.pause(0.001)
		i +=1
	
	return spins, deltaE, E, magnetisation_vector
		
#new_spins, new_delta, new_energy = looper(100000,thirty, 1,1,0,2)
#ax2 = plt.imshow(new_spins)
#plt.savefig('afterlong.png')

def largest_connected_component(nrows, ncols, grid):
    """Find largest connected component of 1s on a grid."""

    def traverse_component(i, j):
        """Returns no. of unseen elements connected to (i,j)."""
        grid[i][j] = -1
        result = 1
     
        # Check all four neighbours
        if i > 0 and (grid[i-1][j]==1):
            result += traverse_component(i-1, j)
        if j > 0 and (grid[i][j-1]==1):
            result += traverse_component(i, j-1)
        if i < len(grid)-1 and (grid[i+1][j]==1):
            result += traverse_component(i+1, j)
        if j < len(grid[0])-1 and (grid[i][j+1]==1):
            result += traverse_component(i, j+1)
        return result

    # Tracks size of largest connected component found
    component_size = 0

    for i in range(nrows):
        for j in range(ncols):
            if (grid[i][j] == 1):
                component_size = max(component_size, traverse_component(i,j))

    return component_size


#Investigating Energy
#Ehalf = np.mean(looper(1000, thirty,1,1,0,0.5)[2])
#E1 = np.mean(looper(1000, thirty,1,1,0,1)[2])
#Ethreehalves = np.mean(looper(1000, thirty,1,1,0,1.5)[2])
#E2 = np.mean(looper(1000, thirty,1,1,0,2)[2])
#Efivehalves = np.mean(looper(1000, thirty,1,1,0,2.5)[2])
#Ethree = np.mean(looper(1000, thirty,1,1,0,3)[2])
#E = np.array([Ehalf, E1, Ethreehalves, E2, Efivehalves, Ethree])
#T = np.array([0.5, 1, 1.5, 2, 2.5, 3])
#print(E)
#print(T)
#plt.plot(T,E)
#plt.show()
#Define a function to calculate the heat capacity over a range of temperatures
#Gives a slightly weird plot, heatcapacity.png
def heat_capacity(temp_low, temp_high, grid, sampling_interval):
	T = np.linspace(temp_low, temp_high, 100)
	heat_cap = np.empty(len(T))
	energy = np.empty(len(T))
	for i in range(len(T)):
		current_energy = looper(sampling_interval, grid, 1, 1, 0, T[i])[2]
		energy[i] = np.mean(current_energy)
		std_dev_energy = np.std(current_energy, dtype=np.float64)
		heat_cap[i] = (np.square(std_dev_energy)/T[i]**2)
	return heat_cap, T, energy

#heat, Temp, energy = heat_capacity(0.5,4,thirty,100000)
#plt.figure(0)
#plt.plot(Temp, heat)
#plt.savefig('heatcaplong.png')
#plt.figure(1)
#plt.plot(Temp, energy)
#plt.savefig('energylong.png')



maghot = looper(10000,ten,1,1,0,10000)[3]
print(np.mean(maghot))

magcold = looper(10000,ten,1,1,0,20)[3]
print(np.mean(magcold))

#primercold = looper(5000,ten,1,1,0,0.5)[0]
#ax3 = plt.imshow(primercold)
#plt.savefig('primerten05.png')
#print(magnetisation(primercold,1))
#print(largest_connected_component(10,10,primercold))

#after = looper(10000,primer,1,1,0,5)[0]
#ax3 = plt.imshow(after)
#plt.savefig('afterfive.png')
#T = np.linspace(0.1, 8, 10)
#magnet = np.empty(len(T))
#for i in range(len(T)):
#	primer = looper(1000,five,1,1,0,T[i])[0]
#	#spin = looper(5000, primer,1,1,0,T[i])[0]
#	magnet[i] = magnetisation(primer,1)
	
	
#plt.figure(0)
#plt.plot(T,magnet)

#plt.show()

