import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from numba import njit
import datetime
print(datetime.datetime.now())

sigma_values = [0.001*(1.35**j) for j in range(13)]

# numerical estimation parameters
n = 201 # must be ODD number for correctly placing the central node
rescale = 250
n_samples = 1000
integration_factor = 2.
MC_samples = int(1e5)
length_factor = 13.
dtau_ML_factor = 5e4
n_ML = n*rescale
T_ML = 10. #note s_0 is around 1., timescale roughly 1. independent of sigma

# CPUs usage
fraction_cores = 0.98
 

def Var_Jf(sigma):
       
    # utils
    corr_length = 1/sigma #note s_0 = 1.
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
    def integrand_J(k, t, alpha):
        sqrt_term = np.sqrt(2 * alpha + k**2)
        numerator = np.cos(k * sigma * t) * np.exp(-sigma * np.abs(t) * sqrt_term)
        denominator = (alpha + k**2) * sqrt_term
        return numerator / denominator
    
    @njit
    def integrand_J_mc(u, t, alpha):
        k = (1 - u) / u
        return np.where(u > 0, integrand_J(k, t, alpha) / (u**2), 0)
    
    @njit
    def compute_integral_J_montecarlo(t, alpha):
        u_samples = np.random.uniform(0, 1, MC_samples)
        integrand_values = integrand_J_mc(u_samples, t, alpha)
        integral_result = np.mean(integrand_values)
        constant_factor = (sigma**2) / (2 * np.pi)
        monte_carlo_estimate = - constant_factor * 2 * integral_result
        return monte_carlo_estimate
    
    @njit
    def Jf_sample():
        r_t = get_r_t()
        time_series = get_events(r_t)
        long_optimum = find_optimum(time_series)
        r_star = reduce_optimum(long_optimum)
        alpha = r_star[mid_n]
        f = r_star - alpha
        J = np.array([compute_integral_J_montecarlo(t, alpha) for t in t_from_point])
        path_integral = np.sum(f*J)*dt
        return path_integral**2
    
    replicas = Parallel(n_jobs=n_cores)(delayed(Jf_sample)() for _ in tqdm(range(n_samples)))
    return np.mean(replicas)


numerical_curve = [] 
reference = []

for sigma in sigma_values:
    print(sigma)
    this = Var_Jf(sigma)
    numerical_curve.append(this)
    print(this)

# Note alpha \approx r_0 = 1
prefactor = 1/(16*8)
Theory = [prefactor*np.power(sigma, 3) for sigma in sigma_values]

Theory = np.array(Theory)
numerical_curve = np.array(numerical_curve)

print('Prefactors')
print(Theory/numerical_curve)


plt.clf()
plt.scatter(sigma_values, numerical_curve, color='black', s = 12)
plt.plot(sigma_values, Theory, color='gray', linewidth = 2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\sigma$', size=14)
plt.legend([r'$\mathbb{E} [ (J_t f^t)^2 ] $', '$\mathbb{E} [(J_t (r^t -r_0 1^t))^2] $'], loc='upper center', fontsize = 14)
plt.tight_layout()
plt.savefig('Jf_sigma.pdf')

plt.clf()
plt.scatter(sigma_values, numerical_curve/Theory, color='black', s = 12)
plt.ylim(0.5*np.min(numerical_curve/Theory), 1.5*np.max(numerical_curve/Theory))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\sigma$', size=14)
plt.ylabel(r'$\mathbb{E} [ (J_t f^t)^2 ] / \mathbb{E} [(J_t (r^t -r_0 1^t))^2] $', size = 14)
plt.tight_layout()
plt.savefig('factor_Jf_sigma.pdf')


np.save('Jf_numerical.npy', numerical_curve)

print(datetime.datetime.now())

