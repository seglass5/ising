"""
ising_data_processing.py
Data is loaded into python, processed and plotted.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Define functions to calculate autocovariance
# Ensure more than two points are included
def check_autocovariance_input(x):
        if len(x) < 2:
            raise ValueError('At least two points required to calculate autocovariance')
# Calculate autocovariance
def get_autocovariance(x):
    check_autocovariance_input(x)
    x_centered = x - np.mean(x)
    return np.correlate(x_centered, x_centered, mode='full')[len(x) - 1:] / len(x)

def main():
    """ Autocovariance, Fig. 2, Fig. 3 and Fig. 4 """
    #Load data
    M = np.load('M_against_steps_homo.npy')
    #M = np.load('M_against_steps_T.npy')
    #M = np.load('M_against_steps.npy')
    
    # Extract time delay
    tau = M[0]
    
    # Calculate autocovariance for each
    autocovariance2 = get_autocovariance(M[1])
    k2 = autocovariance2[0]
    normalised_auto_2 = autocovariance2/k2
    autocovariance4 = get_autocovariance(M[2])
    k4 = autocovariance4[0]
    normalised_auto_4 = autocovariance4/k4
    autocovariance8 = get_autocovariance(M[3])
    k8 = autocovariance8[0]
    normalised_auto_8 = autocovariance8/k8
    autocovariance16 = get_autocovariance(M[4])
    k16 = autocovariance16[0]
    normalised_auto_16 = autocovariance16/k16
    autocovariance32 = get_autocovariance(M[5])
    k32 = autocovariance32[0]
    normalised_auto_32 = autocovariance32/k32
    autocovariance64 = get_autocovariance(M[6])
    k64 = autocovariance64[0]
    normalised_auto_64 = autocovariance64/k64
    #autocovariance128 = get_autocovariance(M[7])
    #k128 = autocovariance128[0]
    #normalised_auto_128 = autocovariance128/k128
    
    # Calculate location of e-folding
    a2 = next(x[0] for x in enumerate(normalised_auto_2) if x[1] < (np.exp(-1)))
    a4 = next(x[0] for x in enumerate(normalised_auto_4) if x[1] < (np.exp(-1)))
    a8 = next(x[0] for x in enumerate(normalised_auto_8) if x[1] < (np.exp(-1)))
    a16 = next(x[0] for x in enumerate(normalised_auto_16) if x[1] < (np.exp(-1)))
    a32 = next(x[0] for x in enumerate(normalised_auto_32) if x[1] < (np.exp(-1)))
    a64 = next(x[0] for x in enumerate(normalised_auto_64) if x[1] < (np.exp(-1)))
    #a128 = next(x[0] for x in enumerate(normalised_auto_128) if x[1] < (np.exp(-1)))
    
    # Print number of iterations at which autocovariance passes efolding threshold
    print(a2)
    print(a4)
    print(a8)
    print(a16)
    print(a32)
    print(a64)
    #print(a128)
    
    # Print corresponding value of autocovariance
    print(normalised_auto_2[a2])
    print(normalised_auto_4[a4])
    print(normalised_auto_8[a8])
    print(normalised_auto_16[a16])
    print(normalised_auto_32[a32])
    print(normalised_auto_64[a64])
    #print(normalised_auto_128[a128])
       
    # Plot autocovariance
    plt.figure(0)
    plt.plot(tau, normalised_auto_2, label='T=0.5')
    plt.plot(tau, normalised_auto_4, label='T=1.5')
    plt.plot(tau, normalised_auto_8, label='T=2')
    plt.plot(tau, normalised_auto_16, label='T=2.5')
    plt.plot(tau, normalised_auto_32, label='T=3')
    plt.plot(tau, normalised_auto_64, label='T=4')
    #plt.plot(tau, normalised_auto_128, label='N=128')
    # Format plot
    plt.legend(loc='best')
    plt.xlim(0,60)
    plt.xlabel('Number of Monte Carlo Iterations')
    plt.ylabel('Autocovariance')
    plt.title('Autocovariance evolution for homogeneous initial states')
    # Save plot
    plt.savefig('Fig3.eps', format='eps', dpi=1000)
    
    """ Thermalised Grids, Fig. 5 """
    # Load Data
    grid_a = np.load('thermal_grid_a.npy')
    grid_b = np.load('thermal_grid_b.npy')
    grid_c = np.load('thermal_grid_c.npy')
    grid_d = np.load('thermal_grid_d.npy')
    grid_e = np.load('thermal_grid_e.npy')
    grid_f = np.load('thermal_grid_f.npy')
    
    # Plot data
    plt.figure(1)
    fig,axes = plt.subplots(2,3)
    axes[0,0].imshow(grid_a)
    axes[0,0].axis('off')
    axes[0,1].imshow(grid_b)
    axes[0,1].axis('off')
    axes[0,2].imshow(grid_c)
    axes[0,2].axis('off')
    axes[1,0].imshow(grid_d)
    axes[1,0].axis('off')
    axes[1,1].imshow(grid_e)
    axes[1,1].axis('off')
    axes[1,2].imshow(grid_f)
    axes[1,2].axis('off')
    # Save Figure
    plt.savefig('Fig5.eps',format='eps',dpi=1000)
    
    """ Energy Temperature, Fig. 6 """
    # Load data
    E = np.load('EnergyTemp.npy')
    # Plot data
    plt.figure(2)
    plt.plot(E[0],E[1],label='N=2')
    plt.plot(E[0],E[2],label='N=4')
    plt.plot(E[0],E[3],label='N=8')
    plt.plot(E[0],E[4],label='N=16')
    plt.plot(E[0],E[5],label='N=32')
    plt.plot(E[0],E[6],label='N=64')
    # Format Plot
    plt.xlabel('$T/J/k_B$')
    plt.xlim(0.5,4.5)
    plt.ylabel('Energy per Spin')
    plt.legend(loc='best')
    plt.title('Energy per Spin against Temperature')
    # Save plot
    plt.savefig('Fig6.eps', format='eps', dpi=1000)
    
    """ Heat Capacity-Temperature, coarse, Fig. 7 """
    # Load data
    C = np.load('CvTemp.npy')
    # Plot
    plt.figure(3)
    plt.plot(C[0],C[1],label='N=2')
    plt.plot(C[0],C[2],label='N=4')
    plt.plot(C[0],C[3],label='N=8')
    plt.plot(C[0],C[4],label='N=16')
    plt.plot(C[0],C[5],label='N=32')
    plt.plot(C[0],C[6],label='N=64')
    # Format
    plt.xlabel('$T/J/k_B$')
    plt.xlim(1.5,3.5)
    plt.ylim(0,3)
    plt.ylabel('Heat Capacity per Spin')
    plt.legend(loc='best')
    plt.title('Heat Capacity per Spin against Temperature')
    # Save plot
    plt.savefig('Fig7.eps', format='eps', dpi=1000)
    
    """ Magnetisation Temperature, Fig. 8 """
    
    MT = np.load('MagTemp.npy')
    plt.figure(4)
    plt.plot(MT[0],MT[1], label='N=2')
    plt.plot(MT[0],MT[2], label='N=4')
    plt.plot(MT[0],MT[3], label='N=8')
    plt.plot(MT[0],MT[4], label='N=16')
    plt.plot(MT[0],MT[5], label='N=32')
    plt.plot(MT[0],MT[6], label='N=64')
    # Format
    plt.legend(loc='best')
    plt.xlim(1.3,4)
    plt.xlabel('$T/J/k_B$')
    plt.ylabel('|M| per spin')
    plt.title('Absolute Magnetisation per Spin against Temperature')
    # Save plot
    plt.savefig('Fig8.eps', format='eps', dpi=1000)

    """ Hysteresis, Fig. 9 """
    # Load data
    H = np.load('HystMag.npy')
    # Plot
    plt.figure(5)
    plt.plot(H[0],H[1],label='T=0.5')
    plt.plot(H[0],H[2],label='T=1.5')
    plt.plot(H[0],H[3],label='T=2.0')
    plt.plot(H[0],H[4],label='T=2.27')
    plt.plot(H[0],H[5],label='T=2.5')
    plt.plot(H[0],H[6],label='T=5')
    # Format plot
    plt.xlabel('Applied Field/$\mu$T')
    plt.ylabel('Absolute Magnetisation per Spin')
    plt.legend(loc='best')
    plt.title('Temperature Dependent Hysteresis Behaviour')
    # Save figure
    plt.savefig('Fig9.eps', format='eps',dpi=1000)
    
    """ Domain Sizes, Fig. 10 and Fig. 11 """
    """ Maximum, Fig. 10 """
    # Load data
    D_max = np.load('dom_temp_max.npy')
    #Plot
    plt.figure(6)
    plt.plot(D_max[0],(D_max[1]/32),label='N=32')
    plt.plot(D_max[0],(D_max[2]/64),label='N=64')
    plt.plot(D_max[0],(D_max[3]/128),label='N=128')
    # Format
    plt.xlim(1.6,3.5)
    plt.xlabel('T/$J/k_B$')
    plt.ylabel('Maximum Fractional Domain Size')
    plt.legend(loc='best')
    plt.title('Maximum Fractional Domain Size against Temperature')
    #Save figure
    plt.savefig('Fig10.eps',format='eps',dpi=1000)
    
    """ Average, Fig. 11 """
    # Load
    D_ave = np.load('dom_temp_average.npy')
    # Plot
    plt.figure(7)
    plt.plot(D_ave[0],(D_ave[1]/32),label='N=32')
    plt.plot(D_ave[0],(D_ave[2]/64),label='N=64')
    plt.plot(D_ave[0],(D_ave[3]/128),label='N=128')
    # Format
    plt.xlim(1.6,3.5)
    plt.xlabel('T/$J/k_B$')
    plt.ylabel('Average Fractional Domain Size')
    plt.legend(loc='best')
    plt.title('Average Fractional Domain Size against Temperature')
    # Save
    plt.savefig('Fig11.eps',format='eps',dpi=1000)
    
    """ Finite Size Scaling, Smoothed Coarse Heat Capacity, Fig. 12 and 14 """
    
    # Reshape temperature
    T = C[0][20:-20][:,np.newaxis]
    
    # Initialise GPR kernel    
    kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-05, noise_level_bounds=(1e-10, 1e+1))
    
    # Apply GPR
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, C[1][20:-20])
    y_mean_2, y_cov_2 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, C[2][20:-20])
    y_mean_4, y_cov_4 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, C[3][20:-20])
    y_mean_8, y_cov_8 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, C[4][20:-20])
    y_mean_16, y_cov_16 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, C[5][20:-20])
    y_mean_32, y_cov_32 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, C[6][20:-20])
    y_mean_64, y_cov_64 = gp.predict(T, return_cov=True)
    
    # Plot data
    plt.figure(8)
    plt.plot(T[:,0], y_mean_2)
    #plt.fill_between(T[:,0], y_mean_2 - np.sqrt(np.diag(y_cov_2)),y_mean_2 + np.sqrt(np.diag(y_cov_2)),alpha=0.5, color='b',label='N=2')
    plt.plot(T[:,0], y_mean_4)
    #plt.fill_between(T[:,0], y_mean_4 - np.sqrt(np.diag(y_cov_4)),y_mean_4 + np.sqrt(np.diag(y_cov_4)),alpha=0.5, color='y',label='N=4')
    plt.plot(T[:,0], y_mean_8)
    #plt.fill_between(T[:,0], y_mean_8 - np.sqrt(np.diag(y_cov_8)),y_mean_8 + np.sqrt(np.diag(y_cov_8)),alpha=0.5, color='g',label='N=8')
    plt.plot(T[:,0], y_mean_16)
    #plt.fill_between(T[:,0], y_mean_16 - np.sqrt(np.diag(y_cov_16)),y_mean_16 + np.sqrt(np.diag(y_cov_16)),alpha=0.5, color='r',label='N=16')
    plt.plot(T[:,0], y_mean_32)
    #plt.fill_between(T[:,0], y_mean_32 - np.sqrt(np.diag(y_cov_32)),y_mean_32 + np.sqrt(np.diag(y_cov_32)),alpha=0.5, color='m',label='N=32')
    plt.plot(T[:,0], y_mean_64)
    #plt.fill_between(T[:,0], y_mean_64 - np.sqrt(np.diag(y_cov_64)),y_mean_64 + np.sqrt(np.diag(y_cov_64)),alpha=0.5, color='c',label='N=64')
    
    # Format
    plt.xlabel('$T/J/k_B$')
    plt.xlim(1.5,3.5)
    plt.ylim(0,3)
    plt.ylabel('Heat Capacity per Spin')
    plt.legend(loc='best')
    plt.title('Heat Capacity per Spin against Temperature')
    # Save figure
    plt.savefig('Fig12.eps', format='eps',dpi=1000)
    
    """ Finite Size Scaling, Smoothed Fine Heat Capacity, Fig. 13 and 14 """
    # Load data
    Cfine = np.load('CvTempFine.npy')
    # Reshape T
    T = Cfine[0][:,np.newaxis]
    
    # Initialise GPR kernel
    kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    
    # Run GPR
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, Cfine[1])
    y_mean_2, y_cov_2 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, Cfine[2])
    y_mean_4, y_cov_4 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, Cfine[3])
    y_mean_8, y_cov_8 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, Cfine[4])
    y_mean_16, y_cov_16 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, Cfine[5])
    y_mean_32, y_cov_32 = gp.predict(T, return_cov=True)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0).fit(T, Cfine[6])
    y_mean_64, y_cov_64 = gp.predict(T, return_cov=True)
    
    # Plot data
    plt.figure(9)
    plt.plot(T[:,0], y_mean_2)
    plt.fill_between(T[:,0], y_mean_2 - np.sqrt(np.diag(y_cov_2)),y_mean_2 + np.sqrt(np.diag(y_cov_2)),alpha=0.5, color='b',label='N=2')
    plt.scatter(T[:,0],Cfine[1],c='b',marker='+')
    plt.plot(T[:,0], y_mean_4)
    plt.fill_between(T[:,0], y_mean_4 - np.sqrt(np.diag(y_cov_4)),y_mean_4 + np.sqrt(np.diag(y_cov_4)),alpha=0.5, color='y',label='N=4')
    plt.scatter(T[:,0],Cfine[2],c='y',marker='+')
    plt.plot(T[:,0], y_mean_8)
    plt.fill_between(T[:,0], y_mean_8 - np.sqrt(np.diag(y_cov_8)),y_mean_8 + np.sqrt(np.diag(y_cov_8)),alpha=0.5, color='g',label='N=8')
    plt.scatter(T[:,0],Cfine[3],c='g',marker='+')
    plt.plot(T[:,0], y_mean_16)
    plt.fill_between(T[:,0], y_mean_16 - np.sqrt(np.diag(y_cov_16)),y_mean_16 + np.sqrt(np.diag(y_cov_16)),alpha=0.5, color='r',label='N=16')
    plt.scatter(T[:,0],Cfine[4],c='r',marker='+')
    plt.plot(T[:,0], y_mean_32)
    plt.fill_between(T[:,0], y_mean_32 - np.sqrt(np.diag(y_cov_32)),y_mean_32 + np.sqrt(np.diag(y_cov_32)),alpha=0.5, color='m',label='N=32')
    plt.scatter(T[:,0],Cfine[5],c='m',marker='+')
    plt.plot(T[:,0], y_mean_64)
    plt.fill_between(T[:,0], y_mean_64 - np.sqrt(np.diag(y_cov_64)),y_mean_64 + np.sqrt(np.diag(y_cov_64)),alpha=0.5,label='N=64')
    plt.scatter(T[:,0],Cfine[6],marker='+')
    
    # Format
    plt.legend(loc='best')
    plt.xlim(2.15,2.55)
    plt.ylim(0,3)
    plt.xlabel('$T/J/k_B$')
    plt.ylabel('Heat Capacity per Spin')
    plt.title('Heat Capacity per Spin against Temperature')
    plt.savefig('Fig13.eps',format='eps',dpi=1000)
    
    # Save smoothed heat capacity data to be find critical temperatures
    smoothed_heat_capacity = np.array([T[1:,0],y_mean_2,y_mean_4,y_mean_8,y_mean_16,y_mean_32,y_mean_64])
    np.save('smoothed_heat_capacity.npy',smoothed_heat_capacity)
    
    #Plot
    plt.figure(10)
    plt.plot(T[:,0], y_mean_2,label='N=2')
    plt.plot(T[:,0], y_mean_4,label='N=4')
    plt.plot(T[:,0], y_mean_8,label='N=8')
    plt.plot(T[:,0], y_mean_16,label='N=16')
    plt.plot(T[:,0], y_mean_32,label='N=32')
    plt.plot(T[:,0], y_mean_64,label='N=64')
    #Format
    plt.legend(loc='best')
    plt.xlim(2.15,2.55)
    plt.ylim(0,3)
    plt.xlabel('$T/J/k_B$')
    plt.ylabel('Heat Capacity per Spin')
    plt.title('Heat Capacity per Spin against Temperature')
    #Save
    plt.savefig('Fig14.eps',format='eps',dpi=1000)
    
    """ Finite Size Scaling, Fig. 15 """
    # Define array of N
    N = np.array([2,4,8,16,32,64])
    # Load
    critical_temperatures = np.load('critical_temperatures.npy')
    
    # Calculate analytical result
    T_inf = 2/(np.log(1+np.sqrt(2)))
    # Finding alpha and nu using linear regression
    y = np.log(critical_temperatures-T_inf)
    x = np.log(N)
    p,V = np.polyfit(x,y,1,cov=True)
    error_gradient = np.sqrt(V[0][0])
    error_intercept = np.sqrt(V[1][1])
    #p[0] -1/nu, ln(alpha) p[1]
    print(p[0],p[1])
    print(error_gradient)
    print(error_intercept)
    
    # Finding Tc inf
    #Linear regression
    p1,V1 = np.polyfit((1/N),critical_temperatures,1,cov=True)
    error_one = np.sqrt(V1[0][0])
    error_intercept_one = np.sqrt(V1[1][1])
    #Tc inf is the intercept
    print(p1[1])
    print(error_intercept_one)
    
    #Plot
    plt.figure(11)
    fig,axes = plt.subplots(1,2)
    #Finding alpha and nu
    axes[0].plot(x, p[0]*x+p[1])
    axes[0].plot(x,y,'+')
    axes[0].set_xlabel('ln(N)')
    axes[0].set_ylabel('ln($T_{C}$(N)-$T_{C}$($\infty$))')
    axes[0].set_title('Finite Size Scaling')
    #Finding Tcinf
    axes[1].plot(1/N,critical_temperatures, '+')
    axes[1].plot(1/N, p1[0]*(1/N)+p1[1])
    axes[1].set_xlabel('$N^{-1/v}$')
    axes[1].set_ylabel('$T_C$')
    axes[1].set_title('Finite Size Scaling')
    #Format and save
    plt.tight_layout()
    plt.savefig('Fig15.eps',format='eps',dpi=1000)
            
    """ Wolff Correlation vs Metropolis """
    # Load
    W = np.load('wolffspeed.npy')
    #Calculate autocovariance
    autocovariance_wolff = get_autocovariance(W[1])
    k_wolff = autocovariance_wolff[0]
    normalised_auto_wolff = autocovariance_wolff/k_wolff
    autocovariance_met = get_autocovariance(W[2])
    k_met = autocovariance_met[0]
    normalised_auto_met = autocovariance_met/k_met
    
    # Calculate and print threshold values
    time_lag_wolff = next(x[0] for x in enumerate(normalised_auto_wolff) if x[1] < (np.exp(-1)))
    time_lag_met = next(x[0] for x in enumerate(normalised_auto_met) if x[1] < (np.exp(-1)))
    print(time_lag_met)
    print(normalised_auto_met[time_lag_met])
    print(time_lag_wolff)
    print(normalised_auto_wolff[time_lag_wolff])
    
    #Plot
    plt.figure(12)
    plt.plot(W[0],normalised_auto_wolff,label='Wolff')
    plt.plot(W[0],normalised_auto_met,label='Metropolis')
    #Format
    plt.xlabel('Number of Iterations')
    plt.xlim(0,400)
    plt.ylabel('Autocovariance')
    plt.legend(loc='best')
    plt.title('Autocovariance of Metropolis and Wolff Algorithms')
    plt.savefig('Fig16.eps',format='eps',dpi=1000)
  
main() 
    