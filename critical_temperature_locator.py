"""
critical_temperature_locator.py
Locate the critical temperatures from saved smoothed_heat_capacity.npy.
Save critical temperatures and N to a numpy array
"""
import numpy as np

def main():
    # Load coarse data for N = 2
    C_coarse = np.load('CvTemp.npy')
    # Elements 0 and 1
    # Load fine data for remaining N
    C_fine= np.load('smoothed_heat_capacity.npy')
    # Initialise array of N values and empty T_crit vector
    N = np.array([2,4,8,16,32,64])
    T_crit = np.empty(len(N))
    # Loop through N>2, find the maximum C and the corresponding temperature
    for i in range(2,len(N)+1):
        a = np.argmax(C_fine[i])
        T_crit[i-1] = (C_fine[0][a])

    a = np.argmax(C_coarse[1])
    T_crit[0] = C_coarse[0][a]
    np.save('critical_temperatures.npy',T_crit)

main()
