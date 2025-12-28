from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags
from scipy.ndimage import shift
from numba import njit
import os
os.chdir("/home/andrea/Desktop/Neuron/")
import datetime
#from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
print(datetime.datetime.now())

fraction_cores = 0.5


# %%

# --- 2. Select one from this top list ---
# Change this index (0 to 4) to pick from the top 5
selected_rank = 2

filename = "top_5_neurons.npz"
loaded_data = np.load(filename, allow_pickle=True)


# 3. Extract the data into your analysis variables
unit_id = loaded_data[f"rank_{selected_rank}_unit_id"]
spikes = loaded_data[f"rank_{selected_rank}_spikes"]
metrics = loaded_data[f"rank_{selected_rank}_metrics"] # [rate, presence, isi]

print(f"Loaded Unit ID: {unit_id} (Rank {selected_rank})")
print(f"Spike Count: {len(spikes)}")
print(f"Original Firing Rate: {metrics[0]:.2f} Hz")


fraction_Data_sigma_estimate = 0.05
fraction_Data_plot = 0.002
init_T_frac = 0.5


# numerical parameters
n = 400
rescale = 400
MC_samples = 5000
n_sigma_estimate = int(n*rescale)
T_ML = 1.
dtau = 0.01


N_sigma = 350

#sigma_list = [0.0005*(1+i) for i in range(N_sigma)] # for rank 0
# sigma_list = [0.0012*(1+i) for i in range(N_sigma)] # for rank 1
sigma_list = [0.0001*(1+i) for i in range(N_sigma)] # for rank 3

n_cores = int(multiprocessing.cpu_count()*fraction_cores)


T_i_Data = 2*spikes[0]-spikes[1]
T_f_Data = 2*spikes[-1]-spikes[-2]

T_i = T_i_Data + (T_f_Data-T_i_Data)*init_T_frac
T_f = T_i + (T_f_Data-T_i_Data)*fraction_Data_sigma_estimate

dt = (T_f - T_i)/n_sigma_estimate
t_grid = np.array([T_i + dt*(i+0.5) for i in range(n_sigma_estimate)])


filtered = []
for this in spikes:
    if this > T_i and this < T_f:
        filtered.append(this)

def get_index (t):
    return int((t-T_i)/dt)


raw = np.array([get_index (t) for t in filtered])
spikes_per_bin = np.array([np.sum(raw==i) for i in range(n_sigma_estimate)])


time_series = 1.* (spikes_per_bin > 0)


# utils
left_border_vec = np.zeros_like(t_grid)
left_border_vec[0] = 1
right_border_vec = np.zeros_like(t_grid)
right_border_vec[-1] = 1
border_terms = (left_border_vec-right_border_vec)/2

# semi-implicit Eulero utils
main_diag = -2 * np.ones(n_sigma_estimate)
main_diag[0] = -1
main_diag[n_sigma_estimate-1] = -1
off_diag = np.ones(n_sigma_estimate)
L_lil = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], n_sigma_estimate, n_sigma_estimate, format='lil')
L = L_lil.tocsc() / dt**2


print(datetime.datetime.now())

print('Saddle point approximation...')


def saddle_point (sigma_prime):
    sigma2 = sigma_prime**2
    
    M = spdiags(np.ones(n_sigma_estimate), 0, n_sigma_estimate, n_sigma_estimate, format='csc') - (dtau / sigma2) * L
    a = M.diagonal(k=-1)
    b = M.diagonal(k=0)
    c = M.diagonal(k=1)
    
    @njit
    def V_opt(r_opt, sigma_prime):
        s_opt = np.log(r_opt)
        Qvar = np.sum(np.power(np.roll(s_opt, -1) - s_opt, 2)[:-1])
        terms_prior = (s_opt[-1]-s_opt[0])/2 +Qvar/(2*np.power(sigma_prime, 2)*dt)
        terms_measurement = -np.sum(time_series*s_opt) + dt*np.sum(r_opt)
        return terms_prior + terms_measurement

    @njit
    def fast_log_det_tridiag(diagonal, off_diag):
        D = np.empty(n_sigma_estimate)
        D[0] = diagonal[0]   
        for i in range(1, n_sigma_estimate):
            D[i] = diagonal[i] - (off_diag[i-1]**2 / D[i-1])
        return np.sum(np.log(D))

    @njit
    def log_det_H_opt(r_opt, sigma_prime):
        m_diag = dt * r_opt + 2.0 / (np.power(sigma_prime, 2) * dt)
        k_off = -1.0 / (np.power(sigma_prime, 2) * dt)
        up_down = np.full(n_sigma_estimate - 1, k_off)
        return fast_log_det_tridiag(m_diag, up_down)

    @njit
    def ln_I (r_opt, sigma_prime):
        return -V_opt(r_opt, sigma_prime) + (n_sigma_estimate/2)*np.log(2*np.pi) -0.5*log_det_H_opt(r_opt, sigma_prime) 
    
    @njit
    def thomas_algorithm(d):
        c_prime = np.zeros(n_sigma_estimate)
        d_prime = np.zeros(n_sigma_estimate)
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        for i in range(1, n_sigma_estimate):
            a_coeff = a[i - 1] 
            m = 1.0 / (b[i] - a_coeff * c_prime[i - 1])
            if i < n_sigma_estimate - 1:
                c_prime[i] = c[i] * m
            else:
                c_prime[i] = 0.0            
            d_prime[i] = (d[i] - a_coeff * d_prime[i - 1]) * m
        x = np.zeros(n_sigma_estimate)
        x[n_sigma_estimate - 1] = d_prime[n_sigma_estimate - 1]  
        for i in range(n_sigma_estimate - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]      
        return x
    
    @njit
    def find_optimum():
        forcing = dtau * (time_series + border_terms) / dt
        x = np.ones_like(time_series) * 0.
        tau = np.float64(0.)
        while tau < T_ML:
            tau += dtau            
            rhs = x + forcing - dtau * np.exp(x)
            x = thomas_algorithm(rhs)
        return np.exp(x)
    
    r_opt = find_optimum()
    s_opt = np.log(r_opt)
    approx = ln_I(r_opt, sigma_prime) -n_sigma_estimate*np.log(sigma_prime) -np.power(sigma_prime, 2)*(T_f-T_i)/8
    return s_opt, approx


## %%

pre_samples = Parallel(n_jobs=n_cores)(delayed(saddle_point)(sigma_list[i]) for i in tqdm(range(N_sigma)))
optima = np.array([i[0] for i in pre_samples])
integral_list = np.array([i[1] for i in pre_samples])

integral_list -= np.max(integral_list)

sigma_posterior = np.exp(integral_list)
sigma_posterior /= np.sum(sigma_posterior)

plt.clf()
plt.plot(sigma_list, sigma_posterior, color = 'black')
plt.xlim(0., max(sigma_list))
this_max = np.max(sigma_posterior)
plt.ylim(0., this_max*1.15)
plt.ylabel('$p(\sigma | D)$', size = 15, rotation = 0, labelpad=25)
plt.xlabel('$\sigma$', size = 14)
plt.tight_layout()
plt.savefig('sigma_posterior.pdf')


np.savetxt("sigma_list.txt", sigma_list, fmt="%.6f")
np.savetxt("sigma_posterior.txt", sigma_posterior, fmt="%.6f")


max_index = np.argmax(sigma_posterior)
sigma_star = sigma_list[max_index]

print(f"The sigma that maximizes the posterior is: {sigma_star}")

print(datetime.datetime.now())

# %%

random_sample = np.random.uniform()

with open('random_sample.txt', 'w') as f:
    print(random_sample, file=f)

sigma = sigma_star

T_i_plot = T_i + random_sample*(T_f_Data-T_i_Data)*fraction_Data_sigma_estimate
T_f_plot = T_i_plot + (T_f_Data-T_i_Data)*fraction_Data_plot
dt = (T_f_plot - T_i_plot)/n_sigma_estimate

t_grid = np.array([T_i_plot + dt*(i+0.5) for i in range(n_sigma_estimate)])


filtered = []
for this in spikes:
    if this > T_i_plot and this < T_f_plot:
        filtered.append(this)

def get_index (t):
    return int((t-T_i_plot)/dt)

raw = np.array([get_index (t) for t in filtered])
spikes_per_bin = np.array([np.sum(raw==i) for i in range(n_sigma_estimate)])

time_series = 1.* (spikes_per_bin > 0)


# semi-implicit Eulero utils
L = L_lil.tocsc() / dt**2

sigma2 = sigma_star**2

M = spdiags(np.ones(n_sigma_estimate), 0, n_sigma_estimate, n_sigma_estimate, format='csc') - (dtau / sigma2) * L
a = M.diagonal(k=-1)
b = M.diagonal(k=0)
c = M.diagonal(k=1)

@njit
def thomas_algorithm(d):
    c_prime = np.zeros(n_sigma_estimate)
    d_prime = np.zeros(n_sigma_estimate)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n_sigma_estimate):
        a_coeff = a[i - 1] 
        m = 1.0 / (b[i] - a_coeff * c_prime[i - 1])
        if i < n_sigma_estimate - 1:
            c_prime[i] = c[i] * m
        else:
            c_prime[i] = 0.0            
        d_prime[i] = (d[i] - a_coeff * d_prime[i - 1]) * m
    x = np.zeros(n_sigma_estimate)
    x[n_sigma_estimate - 1] = d_prime[n_sigma_estimate - 1]  
    for i in range(n_sigma_estimate - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]      
    return x

@njit
def find_optimum():
    forcing = dtau * (time_series + border_terms) / dt
    x = np.ones_like(time_series) * 0.
    tau = np.float64(0.)
    while tau < T_ML:
        tau += dtau            
        rhs = x + forcing - dtau * np.exp(x)
        x = thomas_algorithm(rhs)
    return np.exp(x)

@njit
def reduce_optimum(r_opt):
    reduced = []
    for i in range(n):
        reduced.append(np.mean(r_opt[i*rescale:(i+1)*rescale]))
    return np.array(reduced)


long_optimum = find_optimum()
r_star = reduce_optimum(long_optimum) #actually exp(s_star)
s_star = np.log(r_star)
local_nu = sigma/(2*np.sqrt(r_star))


dt = (T_f_plot - T_i_plot)/n
t_grid = np.array([T_i_plot + dt*(i+0.5) for i in range(n)])

mid_n = int(n/2)
t_from_point = t_grid - (T_i_plot + dt*mid_n)
t_pairs = [(t, tp) for t in t_from_point for tp in t_from_point]




# approximation: using local coefficients
alpha = np.mean(r_star)
nu = sigma/(2*np.sqrt(alpha))
beta = sigma*np.sqrt(alpha)



@njit
def compute_integral_J_montecarlo(t):
    Exp_factor = beta*np.abs(t)
    Cos_factor = beta*t
    theta_max = np.pi / 2.0
    theta = np.random.uniform(0.0, theta_max, n*MC_samples)
    k = np.tan(theta)
    Sqrt_term = np.sqrt(2.0 + k**2)
    Inv_Sqrt_term = 1.0 / Sqrt_term
    Exp_arg = -Exp_factor * Sqrt_term
    Cos_arg = Cos_factor * k
    prefactor = 4.0 / np.pi    
    g_theta = prefactor * Inv_Sqrt_term * np.cos(Cos_arg) * np.exp(Exp_arg)
    integral_estimate = - nu**2 * (theta_max / (n*MC_samples)) * np.sum(g_theta)
    return integral_estimate

@njit
def compute_integral_K1_montecarlo(t, t_prime):
    Exp_factor = beta*(np.abs(t - t_prime) + np.abs(t))
    Cos_factor = beta*t_prime
    theta_max = np.pi / 2.0
    theta = np.random.uniform(0.0, theta_max, MC_samples)
    k = np.tan(theta)
    Denominator = 1.0 / (2.0 + k**2)
    Sqrt_term = np.sqrt(2.0 + k**2)
    Exp_arg = -Exp_factor * Sqrt_term
    Cos_arg = Cos_factor * k
    prefactor = 4.0 / np.pi   
    g_theta = prefactor * Denominator * np.cos(Cos_arg) * np.exp(Exp_arg)
    integral_estimate = nu**3 * (theta_max / MC_samples) * np.sum(g_theta)
    return integral_estimate

@njit
def compute_integral_K2_montecarlo(t, t_prime):
    R1 = np.abs(t - t_prime) + np.abs(t) + np.abs(t_prime)
    R2 = np.abs(t) - np.abs(t_prime)
    R3 = np.abs(t - t_prime)
    theta_max = np.pi / 2.0
    theta = np.random.uniform(0.0, theta_max, MC_samples)
    cos_theta = np.cos(theta)
    sec_theta = 1.0 / (cos_theta + 1e-20)
    A_theta = np.sqrt((sec_theta - 1.0) / 2.0)
    B_theta = np.sqrt((sec_theta + 1.0) / 2.0)
    Exp_arg = -beta * R1  * B_theta
    Cos1_arg = beta * R2 * A_theta
    Cos2_arg = beta * R3 * A_theta
    A_over_tan = (1.0 / (2.0 * np.cos(theta / 2.0))) * np.sqrt(cos_theta)
    g_pre = (1.0 / np.pi) * A_over_tan
    g_theta = g_pre * np.exp(Exp_arg) * np.cos(Cos1_arg) * np.cos(Cos2_arg)
    integral_estimate_0_to_inf = (theta_max / MC_samples) * np.sum(g_theta)
    integral_estimate_neg_inf_to_inf = 2.0 * integral_estimate_0_to_inf
    integral_estimate = nu**3 * integral_estimate_neg_inf_to_inf
    return integral_estimate


print('MonteCarlo integrals ...')

J = Parallel(n_jobs=n_cores)(
    delayed(compute_integral_J_montecarlo)(t) for t in t_from_point)
pre_K1 = Parallel(n_jobs=n_cores)(
    delayed(compute_integral_K1_montecarlo)(t, tp) for t, tp in t_pairs)
K1 = np.array(pre_K1).reshape((n, n))
pre_K2 = Parallel(n_jobs=n_cores)(
    delayed(compute_integral_K2_montecarlo)(t, tp) for t, tp in t_pairs)
K2 = np.array(pre_K2).reshape((n, n))


G1u = nu*np.exp(-sigma*np.sqrt(alpha)*np.abs(t_from_point))

## %%


   
def local_corrections(index):
    
    f = shift(r_star, mid_n-index, mode='nearest') - r_star[index]
    path_integral_J = np.sum(f*J)*dt
    double_path_integral_K1 = np.sum(K1 * np.outer(f, f)) * dt**2
    double_path_integral_K2 = np.sum(K2 * np.outer(f, f)) * dt**2
    
    G_f = np.zeros(n)
    G_ff = np.zeros((n, n))
    for i in range(n):
        f_t_tp = np.roll(f, mid_n-i)
        gap_i = abs(mid_n-i)
        if gap_i > 0:
            G_f[i] = np.sum((G1u*f_t_tp)[gap_i: -gap_i])*dt
        else:
            G_f[i] = np.sum(G1u*f_t_tp)*dt
        for j in range(n):
            f_t_tpp = np.roll(f, mid_n-j)
            gap_j = abs(mid_n - j)
            if gap_i == 0 and gap_j == 0:
                G_ff[i, j] = np.sum(G1u*f_t_tp*f_t_tpp)*dt
            elif gap_j > gap_i:
                G_ff[i, j] = np.sum((G1u*f_t_tp*f_t_tpp)[gap_j: -gap_j])*dt
            else:
                G_ff[i, j] = np.sum((G1u*f_t_tp*f_t_tpp)[gap_i: -gap_i])*dt
    
    spatial_var_correction = -(alpha/2)*(np.sum(J*G_f)*dt + np.sum((K1+K2)*G_ff)*dt**2) 
    
    delta_Gf_f = G_f - f/alpha
    overline_f0_delta_Gf_f = np.zeros(n)
    for i in range(n):
        delta_Gf_f_t_tp = np.roll(delta_Gf_f, mid_n-i)
        gap_i = abs(mid_n-i)
        if gap_i > 0:
            overline_f0_delta_Gf_f[i] = np.sum((G1u*f*delta_Gf_f_t_tp)[gap_i: -gap_i])*dt
        else:
            overline_f0_delta_Gf_f[i] = np.sum(G1u*f*delta_Gf_f_t_tp)*dt
    
    path_var_excess = (alpha/2)*np.sum(J*overline_f0_delta_Gf_f)*dt
    
    x_correction = spatial_var_correction + path_var_excess
    
    path_delta_nu = path_integral_J + double_path_integral_K1 + double_path_integral_K2
    
    delta_nu = np.power(local_nu[index], 2)/9 + path_delta_nu
    adj_nu = local_nu[index]+delta_nu
    delta_mu = -local_nu[index]/2 + x_correction
    
    return delta_mu, adj_nu
    


print(datetime.datetime.now())

print('local corrections ...')

values = Parallel(n_jobs=n_cores)(delayed(local_corrections)(index) for index in tqdm(range(n)))


perturbative_mean = []
perturbative_stdev = []

for this in values:
    perturbative_mean.append(this[0])
    perturbative_stdev.append(np.sqrt(this[1]))

perturbative_mean = np.array(perturbative_mean)
perturbative_stdev = np.array(perturbative_stdev)
ll_stdev = np.sqrt(local_nu)


range_frac = 0.85
min_x = (min(t_grid) + max(t_grid))/2 - range_frac*(max(t_grid)-min(t_grid))/2
max_x = min_x + range_frac*(max(t_grid)-min(t_grid))

plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(t_grid, s_star, color='grey', linestyle='solid', label=r'$s^*$')
ax1.plot(t_grid, s_star + perturbative_mean, color='black', linestyle='solid', 
         label=r'$s^*+\langle x \rangle$')
ax1.set_xlabel('$t [sec]$', size=14)
ax1.set_xlim(min_x, max_x)
ax2 = ax1.twinx()
ax2.plot(t_grid, ll_stdev, color='grey', linestyle='dashed', 
         label=r'$\sqrt{\nu}$')
ax2.plot(t_grid, perturbative_stdev, color='black', linestyle='dashed', 
         label=r'$Std[x]$')
         #label=r'$\sqrt{\langle x^2 \rangle - \langle x \rangle^2}$')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
#ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
           loc='lower center', 
           bbox_to_anchor=(0.5, 1.02), 
           ncol=4, 
           frameon=False)
plt.tight_layout()
plt.savefig('tubes.pdf')



# %%


'''
# IMPORTING DATA

# 1. Define where to save the data (This is important! The data is large)
output_dir = os.path.expanduser('~/allen_data') 
manifest_path = os.path.join(output_dir, "manifest.json")

# 2. Initialize the cache
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# 3. Download/Load a specific session
# (Using session 715093703 as a standard example)
session_id = 715093703
session = cache.get_session_data(session_id)

print(f"Session {session_id} loaded successfully.")

# %%

# 1. Identify available probes in this session
probes = session.probes
print("\n--- Available Probes ---")
print(probes[['description', 'location']])

# 2. Pick the first probe ID
probe_id = probes.index[0]
print(f"\nDownloading/Loading LFP for probe: {probe_id}")

# 3. Load the LFP (This might take a minute to download)
lfp = session.get_lfp(probe_id)

# 'lfp' is an xarray.DataArray (Time x Channel)
print("\n--- LFP Data Shape ---")
print(lfp.shape) # (Time points, Channels)


# %%

# 1. Get the units (neurons) associated with this probe
units = session.units
probe_units = units[units['probe_id'] == probe_id]

# 2. Get spike times for these specific neurons
# This returns a dictionary: {unit_id: [spike_times...]}
all_spike_times = session.spike_times

print(f"Found {len(probe_units)} neurons on Probe {probe_id}")


# %%

good_candidates = probe_units[
    (probe_units['isi_violations'] < 0.5) & 
    (probe_units['presence_ratio'] > 0.9)
]

# Sort by presence_ratio (stability) and then firing_rate (data volume)
# We use 'ascending=False' to get the highest values first
top_neurons = good_candidates.sort_values(
    by=['presence_ratio', 'firing_rate'], 
    ascending=[False, False]
).head(5)

print("--- Top 5 Most Stable Neurons ---")
print(top_neurons[['firing_rate', 'presence_ratio', 'isi_violations']])


# 1. Prepare a dictionary to hold our data
data_to_save = {}

print("--- Packaging Data for Storage ---")
for rank, unit_id in enumerate(top_neurons.index):
    # Get the spikes
    spikes = all_spike_times[unit_id]
    
    # Get the metrics
    metrics = top_neurons.loc[unit_id]
    
    # Store in the dictionary
    # We prefix keys with "rank_X" so it's easy to load later
    data_to_save[f"rank_{rank}_spikes"] = spikes
    data_to_save[f"rank_{rank}_unit_id"] = unit_id
    data_to_save[f"rank_{rank}_metrics"] = metrics[['firing_rate', 'presence_ratio', 'isi_violations']].values

# 2. Save to a single compressed file
filename = "top_5_neurons.npz"
np.savez_compressed(filename, **data_to_save)

'''
