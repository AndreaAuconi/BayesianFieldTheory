import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from numba import njit
import datetime
print(datetime.datetime.now())

# model parameter
sigma = 0.05

# numerical estimation parameters
n = 151 # must be ODD number for correctly placing the central node
rescale = 400
n_samples = 2000
integration_factor = 0.1

# CPUs usage
fraction_cores = 0.5

# utils
length_factor = 15.
dtau_ML_factor = 1e4
n_std_grid = 6.
n_ML = n*rescale
T_ML = 10. #note s_0 is around 1., timescale roughly 1. independent of sigma
corr_length = 1/sigma #note s_0 = 1.
tau_init = 3.
dtau = (1e-4/integration_factor)*(length_factor**2)*(corr_length**2)/(n**2)
dtau_ML = dtau_ML_factor*dtau/np.power(rescale, 2)
T = corr_length*length_factor
dt = T/n
dt_ML = T/n_ML
t_grid = np.array([dt*i for i in range(n)])
t_grid_ML = np.array([dt_ML*i for i in range(n_ML)])
mid_n = int(n/2)
mid_n_ML = int(n_ML/2)
t_from_point = t_grid - dt*mid_n
t_pairs = [(t, tp) for t in t_from_point for tp in t_from_point]
sigma2 = np.power(sigma, 2)
left_border_vec = np.zeros_like(t_grid_ML)
left_border_vec[0] = 1
right_border_vec = np.zeros_like(t_grid_ML)
right_border_vec[-1] = 1
border_terms = (left_border_vec-right_border_vec)/2
n_cores = int(multiprocessing.cpu_count()*fraction_cores)


# implicit Eulero utils
main_diag_ML = -2 * np.ones(n_ML)
main_diag_ML[0] = -1
main_diag_ML[n_ML-1] = -1
off_diag_ML = np.ones(n_ML)
L_lil_ML = spdiags([off_diag_ML, main_diag_ML, off_diag_ML], [-1, 0, 1], n_ML, n_ML, format='lil')
L_ML = L_lil_ML.tocsc() / dt_ML**2
M_ML = spdiags(np.ones(n_ML), 0, n_ML, n_ML, format='csc') - (dtau_ML / sigma2) * L_ML
a_ML = M_ML.diagonal(k=-1)
b_ML = M_ML.diagonal(k=0)
c_ML = M_ML.diagonal(k=1)


@njit
def get_r_t():
    def dW():
        return np.random.normal(0, np.sqrt(dt_ML))
    s = np.zeros(n_ML)
    for j in range(mid_n_ML):
        s[mid_n_ML +j +1] = s[mid_n_ML +j] + sigma*dW() -(sigma2/2)*dt_ML
        s[mid_n_ML -j -1] = s[mid_n_ML -j] + sigma*dW() +(sigma2/2)*dt_ML
    return np.exp(s)

@njit
def get_events(r_t):
    events = []
    for r in r_t:
        events.append(np.random.uniform(0,1) < r*dt_ML)
    return np.array(events)

@njit
def thomas_algorithm_ML(d):
    c_prime = np.zeros(n_ML)
    d_prime = np.zeros(n_ML)
    c_prime[0] = c_ML[0] / b_ML[0]
    d_prime[0] = d[0] / b_ML[0]
    for i in range(1, n_ML):
        a_coeff = a_ML[i-1] 
        m = 1.0 / (b_ML[i] - a_coeff * c_prime[i - 1])
        if i < n_ML - 1:
            c_prime[i] = c_ML[i] * m
        else:
            c_prime[i] = 0.0            
        d_prime[i] = (d[i] - a_coeff * d_prime[i - 1]) * m       
    x = np.zeros(n_ML)
    x[n_ML - 1] = d_prime[n_ML - 1]
    for i in range(n_ML - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]        
    return x


@njit
def find_optimum(time_series):
    forcing = dtau_ML * (time_series + border_terms) / dt_ML
    x = np.ones_like(time_series) * 0.
    tau = np.float64(0.)
    while tau < T_ML:
        tau += dtau_ML            
        rhs = x + forcing - dtau_ML * np.exp(x)
        x = thomas_algorithm_ML(rhs)
    return np.exp(x)

@njit
def reduce_optimum(r_opt):
    reduced = []
    for i in range(n):
        reduced.append(np.mean(r_opt[i*rescale:(i+1)*rescale]))
    return np.array(reduced)

@njit
def f2_sample():
    r_t = get_r_t()
    time_series = get_events(r_t)
    long_optimum = find_optimum(time_series)
    r_star = reduce_optimum(long_optimum)
    alpha = r_star[mid_n]
    f = r_star - alpha
    norm_f = f/alpha
    return norm_f**2

replicas = Parallel(n_jobs=n_cores)(delayed(f2_sample)() for _ in tqdm(range(n_samples)))
numerical = np.mean(replicas, axis = 0)


scaling_factor = 0.5 * sigma**3
Theory = np.array([scaling_factor*(i**2) for i in t_from_point])
time_vec = np.array([np.abs(i) for i in t_from_point])
Underlying = (np.exp(sigma2*time_vec)-1)*(t_from_point>0) \
    + (np.exp(3*sigma2*time_vec)+1-2*np.exp(sigma2*time_vec))*(t_from_point<0)


plt.clf()
plt.scatter(t_from_point, numerical, color='black', s = 10)
plt.plot(t_from_point, Theory, color='gray', linewidth = 2)
plt.plot(t_from_point, Underlying, color='gray', linewidth = 2, linestyle = '--')
plt.xlabel('$t$', size=14)
plt.ylabel(r'$\langle (f_t/\alpha)^2 \rangle $', size = 14)#, rotation = 0)
plt.axvline(0, color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.axvline(corr_length, color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.axvline(-corr_length, color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.xlim(0.4*t_from_point[0], 0.4*t_from_point[-1])
plt.ylim(0, 6 * scaling_factor * corr_length**2)
plt.legend(['Numerical', 'Kalman short-time scaling', 'Underlying process'], loc='upper center')
plt.tight_layout()
plt.savefig('scaling.pdf')


np.save('scaling_numerical.npy', numerical)

print(datetime.datetime.now())

