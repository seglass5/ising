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
    #Number of rows
	n = len(grid)
    #Current energy grid same size as spins
	current_energy = np.empty_like(grid)
    #Delta energy grid same size as spins
	delta_energy = np.empty_like(grid)
	for i in range(0,n):
		for j in range(0,n):
            #Factor of 1/2 to avoid overcounting
			current_energy[i,j] = -magnetic_moment*field*grid[i,j] -(1/2)*exchange_energy*(grid[i,j]*grid[i,(j-1)%n] + grid[i,j]*grid[i,(j+1)%n] + grid[i,j]*grid[(i-1)%n,j] + grid[i,j]*grid[(i+1)%n,j])
	delta_energy = - 2*current_energy
	return delta_energy, current_energy

#Define a function to investigate delta_energy at each position
#Return the flip_matrix
def energy_evaluator(delta_energy, temp):
	#Number of rows in delta_energy
    n = len(delta_energy)
    flip_matrix = np.ones_like(delta_energy)
    for i in range(0,n):
        for j in range(0,n):
            #Generate a new random number each time
            p = np.random.uniform()
            if (delta_energy[i,j] < 0):
                flip_matrix[i,j] = -1
            elif (np.exp((-delta_energy[i,j])/(temp)) > p):
                flip_matrix[i,j] = -1
    return flip_matrix

#One time loop is calculate energy matrix, calculate flip matrix and multiply
#Define a function that takes the required number of time loops and the initial matrix and exchange energy magnetic moment, field and temperature
#And runs it for the specified number of loops, preferably updating plot as it goes
#Define a function to calculate magnetisation
    

def magnetisation(grid, magnetic_moment):
	M = np.abs(magnetic_moment * np.sum(grid))
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
		
#new_spins, new_delta, new_energy, new_mag = looper(100,ten, 1,1,0,2)
#print(new_mag)
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

new_spins, new_delta, new_energy, new_mag = looper(1000,ten, 1,1,0,1)
ax1 = plt.imshow(new_spins)
plt.show()
print(largest_connected_component(10,10,new_spins))
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

#Define a function to calculate the autocorrelation

def autocorrelation(magnetisation_vector, time_delay):
    m_prime = magnetisation_vector - np.mean(magnetisation_vector)
    m_prime_delay = m_prime[time_delay:]
    l = len(m_prime)-time_delay
    A = np.mean(np.multiply(m_prime_delay,m_prime[:l]))
    return A
mmm = np.array([1,4,4])
taus = np.linspace(0, len(mmm),len(mmm)+1, dtype=int)

#print(autocorrelation(mmm,taus[0]))
#print(autocorrelation(mmm,taus[1]))
#a = autocorrelation(magnetisation_vector, time_delay)/autocorrelation(magnetisation_vector,0)
    

#heat, Temp, energy = heat_capacity(0.5,4,thirty,100000)
#plt.figure(0)
#plt.plot(Temp, heat)
#plt.savefig('heatcaplong.png')
#plt.figure(1)
#plt.plot(Temp, energy)
#plt.savefig('energylong.png')

#primer1 = looper(5000, ten, 1,1,0,0.1)[0]
#mag1 = looper(5000,ten,1,1,0,0.1)[3]
#print(np.mean(mag1))

#primer2 = looper(5000, ten, 1,1,0,1)[0]
#mag2 = looper(5000,ten,1,1,0,1)[3]
#print(np.mean(mag2))

#primer3 = looper(5000, ten, 1,1,0,1.5)[0]
#mag3 = looper(5000,ten,1,1,0,1.5)[3]
#print(np.mean(mag3))

#primer4 = looper(5000, ten, 1,1,0,2)[0]
#mag4 = looper(5000,ten,1,1,0,2)[3]
#print(np.mean(mag4))

#primer5 = looper(5000, ten, 1,1,0,3)[0]
#mag5 = looper(5000,ten,1,1,0,3)[3]
#print(np.mean(mag5))

#primer6 = looper(5000, ten, 1,1,0,8)[0]
#mag6 = looper(5000,ten,1,1,0,8)[3]
#print(np.mean(mag6))

#primercold = looper(5000,ten,1,1,0,0.5)[0]
#ax3 = plt.imshow(primercold)
#plt.savefig('primerten05.png')
#print(magnetisation(primercold,1))
#print(largest_connected_component(10,10,primercold))
#print(largest_connected_component(10,10,primerhot))

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

