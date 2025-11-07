import numpy as np 
import pylab as plt
import pandas as pd


#PART A:
#3: 

#using pandas to read file
GWevents = pd.read_csv('gravitationalwaveevents.csv')
#print to validate
print(GWevents)

#create seperate arrays/objects for each column for plotting
DL = GWevents['DL']
DL_err = GWevents['DL_err']
Mtot = GWevents['Mtot']
Mtot_err = GWevents['Mtot_err']

#print just one to validate that it worked 
print(DL)




#plot total mass against distance with error bars 
plt.figure(figsize=(8, 5))
plt.errorbar(DL, Mtot, xerr=DL_err, yerr=Mtot_err, fmt='o',color = 'r' , ecolor='k' ,  capsize=3, label="events") #appropriate colours and capsize
plt.xlabel("Distance (Mpc)")
plt.ylabel("Total Mass (Mtot, $M_\odot$)")
plt.title("Total Mass vs Distance" ,fontsize = 14)
plt.grid(linestyle='--')
plt.legend()


#PART B: 
#1:
waveform_data = pd.read_csv('Observedwaveform.csv')
print(waveform_data)




#2: 
merger_time = 1205951542.153363

#shift the time column so the merger occurs at t = 0 seconds
waveform_data['time_shifted (s)'] = waveform_data['time (s)'] - merger_time
print(waveform_data)



waveform_time = waveform_data['time_shifted (s)'].values
waveform_strain = waveform_data['strain'].values


#strain vs shifted time (s)
plt.figure(figsize=(20, 4))
plt.plot(waveform_time, waveform_strain, label='Observed Strain')
plt.title('strain vs shifted time', fontsize=14)
plt.xlabel('time (s)')
plt.ylabel('strain')
plt.legend()
plt.grid(linestyle = "--")




#filter the data for the region after the merger (t > 0.02)
noise_data = waveform_data[waveform_data['time_shifted (s)'] > 0.02]


#strain vs shifted time (s)
plt.figure(figsize=(20, 4))
plt.plot(noise_data['time_shifted (s)'], noise_data['strain'], label='Observed Strain')
plt.title('strain vs shifted time', fontsize=14)
plt.xlabel('time (s)')
plt.ylabel('strain')
plt.legend()
plt.grid(linestyle = "--")


#calculate the mean and standard deviation of the strain in this new noise window
noise_mean = noise_data['strain'].mean()
noise_std = noise_data['strain'].std()

#display appropriately 
print(f"Noise Mean: {noise_mean:.3e}")
print(f"Noise Standard Deviation: {noise_std:.3e}")


#PART C: 
#1: 
mockdata_waveform_40Msun_1Mpc = pd.read_csv('mockdata_waveform_40Msun_1Mpc.csv')

#strain vs shifted time (s) 
#plot to visualise the mock data
plt.figure(figsize=(20, 4))
plt.plot(mockdata_waveform_40Msun_1Mpc['time (s)'], mockdata_waveform_40Msun_1Mpc['strain'], label='mock data 40Msun 1Mpc')
plt.title('strain vs shifted time', fontsize=14)
plt.xlabel('time (s)')
plt.ylabel('strain')
plt.legend()
plt.grid(linestyle = "--")


t_min = mockdata_waveform_40Msun_1Mpc['time (s)'].min() 
print(f"t_min: {t_min} ")

t_min_o = waveform_data['time_shifted (s)'].min()
print(f"t_min_o: {t_min_o} ")



mockdata_strain = mockdata_waveform_40Msun_1Mpc['strain'].values
mockdata_time = mockdata_waveform_40Msun_1Mpc['time (s)'].values

index = np.where((waveform_time > t_min)&(waveform_time < 0.05))[0]


#strain vs shifted time (s) 
plt.figure(figsize=(20, 4))
plt.plot(waveform_time[index], waveform_strain[index], label='observed data between t_min and 0')
plt.title('strain vs shifted time', fontsize=14)
plt.xlabel('time (s)')
plt.ylabel('strain')
plt.legend()
plt.grid(linestyle = "--")


#Part C:
#2:
reference_waveform_40Msun_1Mpc = pd.read_csv('reference_waveform_40Msun_1Mpc.csv')
ref_time = reference_waveform_40Msun_1Mpc['time (s)'].values
ref_strain = reference_waveform_40Msun_1Mpc['strain'].values

#strain vs shifted time (s) 
#plot to visualise the reference data
plt.figure(figsize=(20, 4))
plt.plot(ref_time, ref_strain, label='reference data for 40Msun and 1Mpc')
plt.title('strain vs shifted time', fontsize=14)
plt.xlabel('time (s)')
plt.ylabel('strain')
plt.legend()
plt.grid(linestyle = "--")



from scipy.interpolate import interp1d


#interpolation object using a reference waveform 
interp_fn = interp1d(ref_time,ref_strain,bounds_error=False)
#interpolate the data waveform
interp_strain = interp_fn(mockdata_time)


#strain vs shifted time (s) 
#plot to visualise the reference data
plt.figure(figsize=(20, 4))
plt.plot(mockdata_time, interp_strain, label='interpolated reference strain')
plt.plot(mockdata_time, mockdata_strain, label='mock data')
plt.title('strain vs shifted time', fontsize=14)
plt.xlabel('time (s)')
plt.ylabel('strain')
plt.legend()
plt.grid(linestyle = "--")


#PART D:
#1:

def generate_waveform(interp_fn, t_ref, M_ref, D_ref, M, D):
   
    #scaled time: t = (M / M_ref) * t_ref
    scaled_time = (M / M_ref) * t_ref
    
    #interpolate the strain at reference time points
    h_ref = interp_fn(t_ref)
    
    #scaled strain: h(t, M, D) = (M / M_ref) * (D_ref / D) * h(t_ref)
    scaled_strain = (M / M_ref) * (D_ref / D) * h_ref

    #create new interpolation function with the scaled time and strain
    interp_fn1 = interp1d(scaled_time,scaled_strain,bounds_error=False)

    #interpolate at the reference time points to match the 
    h = interp_fn1(t_ref)
    
    return scaled_time, h



mockdata_waveform_70Msun_5Mpc = pd.read_csv('mockdata_waveform_70Msun_5Mpc.csv')
mock_time = mockdata_waveform_70Msun_5Mpc['time (s)'].values
mock_strain = mockdata_waveform_70Msun_5Mpc['strain'].values


#PART D:
#2:

#parameters
M_ref = 40  #reference mass (Msun)
D_ref = 1   #reference distance (Mpc)
M = 70      #desired mass (Msun)
D = 5       #desired distance (Mpc)


#generate scaled waveform
t, h = generate_waveform(interp_fn, mock_time, M_ref, D_ref, M, D)

#plot both waveforms for comparison
plt.figure(figsize=(20, 4))
plt.plot(mock_time, mock_strain, label="mock Data (70Msun, 5Mpc)")
plt.plot(mock_time, h, label="Generated Scaled Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Comparison of scaled waveform to mock waveform")
plt.xlim(-0.17,0.06)
plt.legend()
plt.grid()
plt.show()


#PART D:
#3: 

waveform_time = waveform_data['time_shifted (s)'].values
waveform_strain = waveform_data['strain'].values

t_min = waveform_data['time_shifted (s)'].min() 
t_max = waveform_data['time_shifted (s)'].max() 
index = np.where((waveform_time > t_min)&(waveform_time < 0.05))[0]

#define a function to compare waveforms visually
def compare_waveforms(waveform_time, waveform_strain, interp_fn, M_ref, D_ref, M_range, D_range):
    
    plt.figure(figsize=(20, 4))
    
    #plot observed data
    plt.plot(waveform_time, waveform_strain, label="waveform data", linestyle="dashed", linewidth=2, color="black")
    
    #loop over masses and distances
    for M in M_range:
        for D in D_range:
            t, h = generate_waveform(interp_fn, waveform_time[index], M_ref, D_ref, M, D)
            plt.plot(waveform_time[index], h, label=(f"M={M} Msun, D={D:.1f} Mpc"), alpha=0.6)
    
    #format
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.title("scaled template waveforms compared to observed data")
    #plt.legend()
    #plt.xlim(-0.17,0.15)
    plt.grid(linestyle = "--")
    


#define mass and distance ranges for testing
M_range = np.arange(60, 61, 5)  #masses
D_range = np.arange(1000, 2001, 100)  #distances

#compare waveforms
compare_waveforms(waveform_time, waveform_strain, interp_fn, M_ref, D_ref, M_range, D_range)


#define mass and distance ranges for testing
M_range = np.arange(40, 81, 5)  #masses
D_range = np.arange(1000, 1001, 100)  #distances

#compare waveforms
compare_waveforms(waveform_time, waveform_strain, interp_fn, M_ref, D_ref, M_range, D_range)


#Plot 3 (chaning both mass and distance in a smaller range)
M_range = np.arange(70, 80, 2)  #masses
D_range = np.arange(1400, 1601, 100)  #distances
compare_waveforms(waveform_time, waveform_strain, interp_fn, M_ref, D_ref, M_range, D_range)


M = 77
D = 1500

#generate scaled waveform
t, h = generate_waveform(interp_fn, mock_time, M_ref, D_ref, M, D)


#plot both waveforms for comparison
plt.figure(figsize=(20, 4))
plt.plot(waveform_time, waveform_strain, label="waveform data", linestyle="dashed", linewidth=2, color="black")
plt.plot(mock_time, h, label=(f"M={M} Msun, D={D:.1f} Mpc"))
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Comparison of scaled waveform to observed waveform")
plt.xlim(-0.17,0.06)
plt.legend()
plt.grid(linestyle = "--")


def log_likelihood(waveform_strain, waveform_time, interp_fn, M_ref, D_ref, M, D, sigma):
    t,model_strain = generate_waveform(interp_fn, waveform_time_index, M_ref, D_ref, M, D)
    residual = waveform_strain - model_strain
    n = len(model_strain)
    
    return -0.5 * np.sum(residual**2/sigma**2) 
    
#MCMC
def mcmc(waveform_strain, waveform_time, interp_fn, M_ref, D_ref, M_initial, D_initial, sigma, n_steps, mass_step, distance_step):
    M_current = M_initial
    D_current = D_initial
    log_likelihood_current = log_likelihood(waveform_strain, waveform_time, interp_fn, M_ref, D_ref, M_current, D_current, sigma)
    #print(log_likelihood_current)
    M_chain = [M_current]
    D_chain = [D_current]
    
    acceptances = 0  #counter for when aceepted

    for i in range(n_steps):
        
        #propose new parameter values (theta)
        M_proposed = M_current + np.random.normal(0, mass_step)
        D_proposed = D_current + np.random.normal(0, distance_step)
        #print(M_current)
    
        #calculate log-likelihood for proposed parameters
        log_likelihood_proposed = log_likelihood(waveform_strain, waveform_time, interp_fn, M_ref, D_ref, M_proposed, D_proposed, sigma)

        #compute acceptance probability for log space
        log_acceptance_ratio = log_likelihood_proposed - log_likelihood_current
        
        #print(f"Step: {i}, Proposed: {M_proposed}, Current: {M_current}, "f"Log Acceptance Ratio: {log_acceptance_ratio:.2f}")
        
        random = np.log(np.random.rand())
        #print(random)
        
        #accept or reject the proposed parameters
        if random < log_acceptance_ratio:
            M_current = M_proposed
            D_current = D_proposed
            log_likelihood_current = log_likelihood_proposed
            acceptances += 1

        #create the chain for each parameter
        M_chain.append(M_current)
        D_chain.append(D_current)


    acceptance_rate = acceptances / n_steps
    print(f"Acceptance rate: {acceptance_rate:.2f}")
    return np.array(M_chain), np.array(D_chain)

#run the MCMC algorithm
M_initial = 75
D_initial = 1500
n_steps = 1000000
mass_step = 0.01
distance_step = 0.1
sigma = noise_mean #set to the noise value we calculated earlier

#only require the data between index, this also avoids any erros with interpolation
index = np.where((waveform_time > t_min)&(waveform_time < 0.05))[0]

waveform_strain_index = waveform_strain[index]
waveform_time_index = waveform_time[index]

M_chain, D_chain = mcmc(waveform_strain_index, waveform_time_index, interp_fn, M_ref, D_ref, M_initial, D_initial, sigma, n_steps, mass_step, distance_step)


#plot the results appropriately

#mass chain
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(M_chain, label="Mass Chain")
plt.xlabel("step number")
plt.ylabel("Mass")
plt.ylim(76.98,77.02)
plt.legend()

#distance chain
plt.subplot(2, 2, 2)
plt.plot(D_chain, label="Distance Chain")
plt.xlabel("step number")
plt.ylabel("Distance")
plt.ylim(1660,1670)
plt.legend()

#include burn in
#mass histogram 
plt.subplot(2, 2, 3)
plt.hist(M_chain[10000:], bins=100, label="Posterior Distribution")
plt.xlabel("Mass")
plt.ylabel("Density")
plt.legend()

#distance histogram
plt.subplot(2, 2, 4)
plt.hist(D_chain[10000:], bins=100, label="Posterior Distribution")
plt.xlabel("Distance")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()


#rolling mean function
def rolling_mean(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

#generate multiple MCMC chains with different initial guesses
def run_multiple_chains(initial_guesses, n_steps, mass_step):
    M_chains = []
    for i in initial_guesses:
        M_chain_test,_ = mcmc(waveform_strain_index, waveform_time_index, interp_fn, M_ref, D_ref, M_initial, D_initial, sigma, n_steps, mass_step, distance_step)
        M_chains.append(M_chain_test)
    return M_chains

def convergence_test(M_chains, window):
    n_chains = len(M_chains)
    rolling_means = [rolling_mean(M_chain, window) for M_chain in M_chains]
    
    #compute differences between rolling means of each pair of chains
    differences = []
    for i in range(n_chains - 1):
        for j in range(i + 1, n_chains):
            diff = np.abs(rolling_means[i] - rolling_means[j][:len(rolling_means[i])])
            differences.append(diff)
    
    #average the differences for all pairs
    mean_difference = np.mean(differences, axis=0)
    return mean_difference

#initial guesses for mass parameter
initial_guesses = [50, 60, 77, 90, 100]  
n_steps = 10000
mass_step = 0.01
distance_Step = 0.1
window_size = 1000

#run the chains for each mass value
M_chains = run_multiple_chains(initial_guesses, n_steps, mass_step)

#calculate difference in means
mean_difference = convergence_test(M_chains, window_size)

#plot results
plt.figure(figsize=(8, 5))
plt.plot(mean_difference, label="Mean Difference of Rolling Means")
plt.axhline(0, color="red", linestyle="--", label="Target: Convergence")
plt.xlabel("Step")
plt.ylabel("Difference" )
plt.legend()
plt.title("Convergence Test")
plt.show()


#compute statistics
import statistics

#for mass value
M_median = statistics.median(M_chain)
lower_limit_M = np.percentile(M_chain, 5)
upper_limit_M = np.percentile(M_chain, 95)

#for distance value
D_median = statistics.median(D_chain)
lower_limit_D = np.percentile(D_chain, 5)
upper_limit_D = np.percentile(D_chain, 95)

#print results
print(f"Median of M: {M_median:.4f}")
print(f"90% Credible Interval for M: ({lower_limit_M:.4f}, {upper_limit_M:.4f})")

print(f"Median of D: {D_median:.4f}")
print(f"90% Credible Interval for D: ({lower_limit_D:.4f}, {upper_limit_D:.4f})")
print()

#calculate mean mass and mean distance making sure to account for burn in.
burn_in = 10000
best_mass = np.mean(M_chain[burn_in:])
print(f"Best Estimate for Total Distance: {best_mass:.2f} Msun")

best_distance = np.mean(D_chain[burn_in:])
print(f"Best Estimate for Total Mass: {best_distance:.2f} Mpc")

model_time, model_strain = generate_waveform(interp_fn, waveform_time_index, M_ref, D_ref, best_mass, best_distance)

#plot both waveforms for comparison
plt.figure(figsize=(20, 4))
plt.plot(waveform_time_index, waveform_strain_index, label="waveform data", linestyle="dashed", linewidth=2, color="black")
plt.plot(waveform_time_index, model_strain, label=(f"M={best_mass} Msun, D={best_distance} Mpc"))
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Comparison of scaled waveform to observed waveform")
plt.xlim(-0.17,0.06)
plt.legend()
plt.grid(linestyle = "--")


def chirp_mass(total_mass, q):
    return total_mass * (q**(3/5)) / ((1 + q)**(6/5))


def individual_masses(total_mass, q):
    M1 = total_mass / (1 + q)
    M2 = q * total_mass / (1 + q)
    return M1, M2

total_mass = best_mass  
q = 1  #equal mass binaries
chirp_mass = chirp_mass(total_mass, q)
print(f"Chirp mass: {chirp_mass:.2f} Msun")

M1, M2 = individual_masses(total_mass,q)
print(f"M1: {M1:.2f} Msun, M2: {M2:.2f} Msun")


def estimate_period(time, strain):
   
    #peak amplitude index
    peak_index = np.argmax(np.abs(strain))  # Find index of peak amplitude
    peak_time = time[peak_index]
    
    #define window around the peak
    window = 0.008  # Adjust window size as needed
    window_index = (time > (peak_time - window)) & (time < (peak_time + window))
    #time and strain for only the part of the waveform inside the window
    window_time = time[window_index]
    window_strain = strain[window_index]
    
    #find zero-crossings
    zero_crossings = np.where(np.diff(np.sign(window_strain)))[0]
    if len(zero_crossings) > 1:
        #calculate time differences between consecutive zero-crossings
        crossing_times = window_time[zero_crossings]
        periods = np.diff(crossing_times)
        average_period = np.mean(periods)
    
    return average_period

period = estimate_period(waveform_time_index, waveform_strain_index)
print(f"Estimated period: {period:.5f} seconds")

plt.figure(figsize=(20, 4))
plt.plot(waveform_time_index, waveform_strain_index, label="Observed waveform")
plt.axvline(x=waveform_time_index[np.argmax(np.abs(waveform_strain_index))], color='r', linestyle='--', label="peak amplitude")
plt.axvline(x=waveform_time_index[np.argmax(np.abs(waveform_strain_index))]+0.008, color='b', linestyle='--', label="window bounds")
plt.axvline(x=waveform_time_index[np.argmax(np.abs(waveform_strain_index))]-0.008, color='b', linestyle='--')
plt.legend()
plt.title("Observed Waveform with Peak Amplitude")
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.xlim(-0.17,0.06)
plt.legend()
plt.grid(linestyle = "--")


def seperation(M1,M2,omega):
    #constants
    G = 6.674e-11
    Msun = 1.989e30 #convert to kg for correct units
    return ((G*(M1+M2)*Msun)/omega**2)**(1/3)

omega = np.pi/period
R = seperation(M1,M2,omega)
R_km = R/1000
print(f"Oribital seperation, R = {R_km:.2f}km")
