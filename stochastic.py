import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.special import erf
from scipy.ndimage import shift
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from numba import njit
import datetime
print(datetime.datetime.now())

# model parameter
sigma = 0.05

# numerical estimation parameters
n = 401 # must be ODD number for correctly placing the central node
integration_factor = 0.1
MC_samples = int(3e5)
exclude_path = False

# CPUs usage
fraction_cores = 0.5

# utils
length_factor = 15.
statistical_factor = 200.
rescale = 500
dtau_ML_factor = 1e4
n_std_grid = 6.
n_samples = 80
half_n_samples = n_samples/2
n_ML = n*rescale
T_ML = 10. #note s_0 is around 1., timescale roughly 1. independent of sigma
corr_length = 1/sigma #note s_0 = 1. 
tau_init = 3.
dtau = (1e-4/integration_factor)*(length_factor**2)*(corr_length**2)/(n**2)
dtau_ML = dtau_ML_factor*dtau/np.power(rescale, 2)
sampling = int(0.1/dtau)
T = corr_length*length_factor
dt = T/n
dt_ML = T/n_ML
t_grid = np.array([dt*i for i in range(n)])
t_grid_ML = np.array([dt_ML*i for i in range(n_ML)])
mid_n = int(n/2)
mid_n_ML = int(n_ML/2)
t_from_point = t_grid - dt*mid_n
t_pairs = [(t, tp) for t in t_from_point for tp in t_from_point]
sigma2 = np.power(sigma,2)
noise_factor = np.sqrt(2*dtau/dt)
left_border_vec = np.zeros_like(t_grid_ML)
left_border_vec[0] = 1
right_border_vec = np.zeros_like(t_grid_ML)
right_border_vec[-1] = 1
border_terms = (left_border_vec-right_border_vec)/2
n_cores = int(multiprocessing.cpu_count()*fraction_cores)
dx_grid = 2*n_std_grid*np.sqrt(sigma/2)/n_samples
x_grid = np.array([(i+0.5-n_samples/2)*dx_grid for i in range(n_samples)])
split_sim = 10
T_MC = statistical_factor*(1/(split_sim*n_cores))*8/np.power(sigma/9, 2)
calibration_ratio = 0.2

# semi-implicit Eulero utils
main_diag = -2 * np.ones(n)
main_diag[0] = -1
main_diag[n-1] = -1
off_diag = np.ones(n)
L_lil = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], n, n, format='lil')
L = L_lil.tocsc() / dt**2
M = spdiags(np.ones(n), 0, n, n, format='csc') - (dtau / sigma2) * L
main_diag_ML = -2 * np.ones(n_ML)
main_diag_ML[0] = -1
main_diag_ML[n_ML-1] = -1
off_diag_ML = np.ones(n_ML)
L_lil_ML = spdiags([off_diag_ML, main_diag_ML, off_diag_ML], [-1, 0, 1], n_ML, n_ML, format='lil')
L_ML = L_lil_ML.tocsc() / dt_ML**2
M_ML = spdiags(np.ones(n_ML), 0, n_ML, n_ML, format='csc') - (dtau_ML / sigma2) * L_ML
a_Langevin = M.diagonal(k=-1)
b_Langevin = M.diagonal(k=0)
c_Langevin = M.diagonal(k=1)
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
def thomas_algorithm_Langevin(d):
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    c_prime[0] = c_Langevin[0] / b_Langevin[0]
    d_prime[0] = d[0] / b_Langevin[0]
    for i in range(1, n):
        a_coeff = a_Langevin[i - 1] 
        m = 1.0 / (b_Langevin[i] - a_coeff * c_prime[i - 1])
        if i < n - 1:
            c_prime[i] = c_Langevin[i] * m
        else:
            c_prime[i] = 0.0            
        d_prime[i] = (d[i] - a_coeff * d_prime[i - 1]) * m
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]  
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]      
    return x

@njit
def find_optimum(time_series):
    if exclude_path:
        return np.ones_like(time_series)
    else:
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

r_t = get_r_t()
time_series = get_events(r_t)


print('Find ML...')

long_optimum = find_optimum(time_series)
r_star = reduce_optimum(long_optimum) #actually exp(s_star)
alpha = r_star[mid_n]
nu = sigma/(2*np.sqrt(alpha)) 
sqrt_nu = np.sqrt(nu)

Gauss = []
for x in x_grid:
    up = (1/2)*(1 + erf((x+dx_grid/2)/np.sqrt(2*nu)))
    down = (1/2)*(1 + erf((x-dx_grid/2)/np.sqrt(2*nu)))
    Gauss.append((up-down)/dx_grid)
Gauss = np.array(Gauss)
Gauss /= np.sum(Gauss)


print(datetime.datetime.now())

plt.clf()
plt.plot(t_grid_ML, r_t, color = 'gray', linewidth = 0.8)
plt.plot(t_grid_ML, long_optimum, color = 'black', linewidth = 1.)
plt.axvline(t_grid_ML[mid_n_ML], color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.ylim(0, 1.5*np.max(long_optimum))
plt.xlim(t_grid[0], t_grid[-1])
plt.legend(['$r$', '$\exp(s^*)$'], fontsize = 14)
plt.xlabel('$t$', size = 14, rotation = 0)
plt.tight_layout()
plt.savefig('ML.pdf')


np.save('t_grid_ML.npy', t_grid_ML)
np.save('r_t.npy', r_t)
np.save('long_optimum.npy', long_optimum)
np.save('time_series.npy', time_series)


## %%

@njit
def OU_init():
    s = np.random.normal(0, sqrt_nu)
    vec = []
    for _ in range(n):
        s = s*np.exp(-dt*sigma*np.sqrt(alpha)) + sigma*np.random.normal(0, np.sqrt(dt))
        vec.append(s)
    return np.array(vec)

@njit
def noise_calibration():
    def space_time_noise():
        return np.random.normal(0, noise_factor, n)
    discrete_Var = 0.
    discrete_Mean = 0.
    n_samples = 0 
    x = OU_init()
    tau = np.float64(0.)
    i = 0
    while tau < T_MC*calibration_ratio:
        tau += dtau
        rhs = x - dtau * alpha * x + space_time_noise()
        x = thomas_algorithm_Langevin(rhs)      
        i += 1
        if i == sampling:
            i = 0
            discrete_Mean += x[mid_n]
            discrete_Var += x[mid_n]**2
            n_samples += 1
    discrete_Mean /= n_samples
    discrete_Var /= n_samples
    discrete_Var -= discrete_Mean**2
    return discrete_Var / nu

print(datetime.datetime.now())

print('noise calibration...')
cal_replicas = Parallel(n_jobs=n_cores)(delayed(noise_calibration)() for _ in tqdm(range(split_sim*n_cores)))
Var_ratio = np.mean(cal_replicas)
noise_correction = 1/np.sqrt(Var_ratio)
adjusted_noise_factor = noise_factor*noise_correction
print(noise_correction)
print(datetime.datetime.now())

## %%

@njit
def Langevin_statistics():
    def space_time_noise():
        return np.random.normal(0, adjusted_noise_factor, n)
    vec = np.zeros(n_samples)
    x = OU_init()
    tau = np.float64(0.)
    i = 0
    while tau < T_MC:
        tau += dtau
        rhs = x - dtau * r_star * (np.exp(x) - 1) + space_time_noise()
        x = thomas_algorithm_Langevin(rhs)      
        i += 1
        if i == sampling:
            i = 0
            if tau > tau_init:
                x_cont = x[mid_n]/dx_grid +half_n_samples
                if x_cont > 0.:
                    j = int(x_cont)
                    if j < n_samples:
                        vec[j] += 1
    return vec

@njit
def integrand_J(k, t, alpha, sigma):
    sqrt_term = np.sqrt(2 * alpha + k**2)
    numerator = np.cos(k * sigma * t) * np.exp(-sigma * np.abs(t) * sqrt_term)
    denominator = (alpha + k**2) * sqrt_term
    return numerator / denominator

@njit
def integrand_J_mc(u, t):
    k = (1 - u) / u
    return np.where(u > 0, integrand_J(k, t, alpha, sigma) / (u**2), 0)

@njit
def compute_integral_J_montecarlo(t):
    u_samples = np.random.uniform(0, 1, 10*MC_samples)
    integrand_values = integrand_J_mc(u_samples, t)
    integral_result = np.mean(integrand_values)
    constant_factor = (sigma**2) / (2 * np.pi)
    monte_carlo_esimate = - constant_factor * 2 * integral_result
    return monte_carlo_esimate

@njit
def integrand_K_montecarlo(theta, phi, psi, t, tp):
    k = 1 / np.tan(theta)
    u = np.tan(phi)**2
    up = np.tan(psi)**2
    jacobian_k = (1 / np.sin(theta))**2
    jacobian_u = 2 * np.tan(phi) * (1 / np.cos(phi))**2
    jacobian_up = 2 * np.tan(psi) * (1 / np.cos(psi))**2  
    term1 = 1.0 / (np.sqrt(u * up) * (alpha + k**2))
    term2 = np.cos(k * sigma * (t - tp))
    exp1_arg = -alpha * (u + up) - (sigma**2 / 4) * (t**2/u + tp**2/up)
    exp2_arg = -(alpha + k**2) * np.abs(u - up)  
    return term1 * term2 * np.exp(exp1_arg + exp2_arg) * jacobian_k * jacobian_u * jacobian_up

@njit
def compute_integral_K_montecarlo(t_val, tp_val):
    theta_samples = np.random.uniform(0, np.pi, MC_samples)
    phi_samples = np.random.uniform(0, np.pi/2, MC_samples)
    psi_samples = np.random.uniform(0, np.pi/2, MC_samples)
    integrand_values = integrand_K_montecarlo(theta_samples, phi_samples, psi_samples, t_val, tp_val)
    volume_transformed = np.pi * (np.pi/2) * (np.pi/2)
    monte_carlo_estimate = volume_transformed * np.mean(integrand_values)  
    return monte_carlo_estimate

@njit
def integrand_Q_montecarlo(u_t_x, u_t_y, k_t_z, t, tprime):
    # u: [-inf, 0], u': [-inf, u], k: [-inf, inf]
    u = -np.tan(np.pi/2 * u_t_x)
    u_prime = u - np.tan(np.pi / 2 * u_t_y)
    k = np.tan(np.pi * (k_t_z - 0.5))
    jac_u = (np.pi/2) * (1 / np.cos(np.pi/2 * u_t_x))**2
    jac_u_prime = (np.pi/2) * (1 / np.cos(np.pi/2 * u_t_y))**2
    jac_k = np.pi * (1 / np.cos(np.pi * (k_t_z - 0.5)))**2
    g_minus_t_u = (sigma / (2 * np.sqrt(-np.pi * u))) * np.exp(alpha * u + (sigma**2 * t**2) / (4 * u))
    g_t_minus_tprime_u_minus_uprime = (sigma / (2 * np.sqrt(np.pi * (u - u_prime)))) * np.exp(-alpha * (u - u_prime) - (sigma**2 * (t - tprime)**2) / (4 * (u - u_prime)))
    c_tprime_uprime = (sigma / (2 * np.pi)) * (np.cos(k * sigma * tprime) / (alpha + k**2)) * np.exp(-(alpha + k**2) * np.abs(u_prime))
    integrand = 2 * g_minus_t_u * g_t_minus_tprime_u_minus_uprime * c_tprime_uprime
    return integrand * jac_u * jac_u_prime * jac_k
    
@njit
def compute_integral_Q_montecarlo(t_val, tp_val):
    u_t_x_samples = np.random.uniform(0.0, 1.0, MC_samples)
    u_t_y_samples = np.random.uniform(0.0, 1.0, MC_samples)
    k_t_z_samples = np.random.uniform(0.0, 1.0, MC_samples)
    integrand_values = integrand_Q_montecarlo(u_t_x_samples, u_t_y_samples, k_t_z_samples, t_val, tp_val)
    monte_carlo_estimate = np.mean(integrand_values)  
    return monte_carlo_estimate

beta = sigma*np.sqrt(alpha)

@njit
def integrand_J_alpha(k, tp):
    sqrt_term = np.sqrt(2 + k**2)
    abs_tp = np.abs(tp)
    exponential_term = np.exp(-beta * abs_tp * sqrt_term)
    term1 = k * tp * np.sin(beta * k * tp)
    term2 = sqrt_term * abs_tp * np.cos(beta * k * tp)
    numerator_content = term1 + term2 
    numerator = numerator_content * exponential_term
    denominator = (1 + k**2) * sqrt_term
    return numerator / denominator

@njit
def mc_transf_J_alpha(u, tp):
    k = (1.0 - u) / u
    return np.where(u > 0, integrand_J_alpha(k, tp) / (u**2), 0.0)

@njit
def integral_J_alpha(tp):
    u_samples = np.random.uniform(0.0, 1.0, MC_samples)
    integrand_values = mc_transf_J_alpha(u_samples, tp)
    integral_result_half = np.mean(integrand_values)
    const_coeff = np.power(nu,3) *(2/np.pi)
    monte_carlo_esimate = const_coeff * 2.0 * integral_result_half   
    return monte_carlo_esimate


print('Path integrals...')

J = Parallel(n_jobs=n_cores)(
    delayed(compute_integral_J_montecarlo)(t) for t in t_from_point)
pre_K = Parallel(n_jobs=n_cores)(
    delayed(compute_integral_K_montecarlo)(t, tp) for t, tp in t_pairs)
K = np.array(pre_K).reshape((n, n))
constant_factor = (sigma**3) / (8 * np.pi**2)
K = constant_factor * K
pre_Q = Parallel(n_jobs=n_cores)(
    delayed(compute_integral_Q_montecarlo)(t, tp) for t, tp in t_pairs)
Q = np.array(pre_Q).reshape((n, n))
J_alpha = Parallel(n_jobs=n_cores)(
    delayed(integral_J_alpha)(t) for t in t_from_point)

f = r_star - alpha
path_integral = np.sum(f*J)*dt
double_path_integral_K = np.sum(K * np.outer(f, f)) * dt**2
double_path_integral_Q = np.sum(Q * np.outer(f, f)) * dt**2


term1 = -(3/8)*nu*np.power(f/alpha, 2)
term2 = -(f/alpha) * np.sum(f*J)*dt + f * np.sum(f*J_alpha)*dt

def f_shift (i):
    return shift(f, shift=int(n/2)-i, cval=0.0)
term3 = np.array([np.sum(J*(f_shift(i)-f))*dt for i in range(n)])


spatial_var = term1 + term2 + term3

G_t = nu * np.exp(-sigma*np.sqrt(alpha)*np.abs(t_from_point))
x_correction = -(alpha/2)*np.sum(G_t*spatial_var)*dt


path_delta_nu = path_integral + double_path_integral_K + double_path_integral_Q

print(path_integral/np.power(nu, 2))
print(double_path_integral_K/np.power(nu, 2))
print(double_path_integral_Q/np.power(nu, 2))
print(x_correction/np.power(nu, 2))

plt.clf()
plt.plot(t_grid, f*J, color='black', linewidth = 1.5)
plt.xlabel('$t$', size=14)
plt.ylabel(r'$f_t \, J_t$', size = 14, rotation = 0)
plt.axvline(t_grid[mid_n], color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.axhline(0., color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.xlim(t_grid[0], t_grid[-1])
plt.tight_layout()
plt.savefig('path_integral.pdf')

plt.clf()
plt.plot(t_grid, G_t*spatial_var, color='black', linewidth = 1.5)
plt.xlabel('$t$', size=14)
plt.ylabel(r'$u_t \, G_t$', size = 14, rotation = 0)
plt.axvline(t_grid[mid_n], color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.axhline(0., color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.xlim(t_grid[0], t_grid[-1])
plt.tight_layout()
plt.savefig('spatial_var_correction_integral.pdf')

plt.clf()
plt.imshow(K * np.outer(f, f), extent=[t_grid[0], t_grid[-1], t_grid[0], t_grid[-1]], cmap='Greys', origin='lower', interpolation='None')
plt.xlabel('$t$', size=14)
plt.ylabel("$t'$", size=14, rotation = 0)
plt.tight_layout()
plt.savefig('path_double_integral_K.pdf')

plt.clf()
plt.imshow(Q * np.outer(f, f), extent=[t_grid[0], t_grid[-1], t_grid[0], t_grid[-1]], cmap='Greys', origin='lower', interpolation='None')
plt.xlabel('$t$', size=14)
plt.ylabel("$t'$", size=14, rotation = 0)
plt.tight_layout()
plt.savefig('path_double_integral_Q.pdf')


np.save('t_grid.npy', t_grid)
np.save('f.npy', f)
np.save('J.npy', J)
np.save('K.npy', K)
np.save('Q.npy', Q)
np.save('J_alpha.npy', J_alpha)

print(datetime.datetime.now())

print('statistics...')
replicas = Parallel(n_jobs=n_cores)(delayed(Langevin_statistics)() for _ in tqdm(range(split_sim*n_cores)))
statistics = np.sum(replicas, axis = 0)
statistics /= np.sum(statistics)
numerical = statistics - Gauss
print(datetime.datetime.now())


z = x_grid/sqrt_nu
delta_nu = np.power(nu, 2)/9 + path_delta_nu
adj_nu = nu+delta_nu
delta_mu = -(nu + path_delta_nu)/2 + x_correction

adj_sqrt_nu = np.sqrt(adj_nu)

adj_Gauss = []
for x in x_grid:
    up = (1/2)*(1 + erf((x+dx_grid/2-delta_mu)/np.sqrt(2*adj_nu)))
    down = (1/2)*(1 + erf((x-dx_grid/2-delta_mu)/np.sqrt(2*adj_nu)))
    adj_Gauss.append((up-down)/dx_grid)
adj_Gauss = np.array(adj_Gauss)
adj_Gauss /= np.sum(adj_Gauss)

adj_z = (x_grid-delta_mu)/np.sqrt(adj_nu)
k3 = -np.sqrt(nu)/3
third_order = adj_Gauss*(np.power(adj_z, 3) -3*adj_z)*k3/6.
corrected = adj_Gauss + third_order
corrected /= np.sum(corrected)
Theory = corrected - Gauss

plt.clf()
plt.scatter(x_grid, numerical, color='black', s = 15)
plt.plot(x_grid, Theory, color='gray', linewidth = 2)
plt.plot(x_grid, adj_Gauss - Gauss, color='gray', linestyle='--', linewidth = 2)
plt.xlim(0.8*x_grid[0], 0.8*x_grid[-1])
plt.xlabel(r'$x$', size=14)
plt.ylabel(r'$\delta p$', size=14, rotation = 0)
plt.legend(['numerical', '3rd order', '2nd order'])
plt.tight_layout()
plt.savefig('dp_orders.pdf')

np.save('x_grid.npy', x_grid)
np.save('numerical.npy', numerical)
np.save('theory.npy', Theory)
np.save('adj_Gauss.npy', adj_Gauss)
np.save('Gauss.npy', Gauss)


num_nu = np.sum(Gauss*np.power(x_grid, 2))
num_delta_mu = np.sum(statistics*x_grid)
num_delta_nu = np.sum(statistics*np.power(x_grid, 2)) -np.power(num_delta_mu, 2) -num_nu


print('numerical ratio')
print((num_nu-nu)/nu)

print('delta_mu')
print(delta_mu)
print(num_delta_mu)

print('delta_nu')
print(delta_nu)
print(num_delta_nu)

print('x3')
num_x_3 = np.sum(statistics*np.power(x_grid, 3))
print(-np.power(nu, 2.)*11/6)
print(num_x_3)

print('k3')
num_k_3 = num_x_3 -3*(num_nu+num_delta_nu+num_delta_mu**2)*num_delta_mu +2*num_delta_mu**3
num_k_3 /= np.power(num_nu+num_delta_nu+num_delta_mu**2, 3/2)
print(k3)
print(num_k_3)


NP_delta_nu = np.power(nu, 2)/9
NP_adj_nu = nu+NP_delta_nu
NP_delta_mu = -nu/2
NP_adj_z = (x_grid-NP_delta_mu)/np.sqrt(NP_adj_nu)

NP_adj_Gauss = []
for x in x_grid:
    up = (1/2)*(1 + erf((x+dx_grid/2-NP_delta_mu)/np.sqrt(2*NP_adj_nu)))
    down = (1/2)*(1 + erf((x-dx_grid/2-NP_delta_mu)/np.sqrt(2*NP_adj_nu)))
    NP_adj_Gauss.append((up-down)/dx_grid)
NP_adj_Gauss = np.array(NP_adj_Gauss)
NP_adj_Gauss /= np.sum(NP_adj_Gauss)

NP_third_order = NP_adj_Gauss*(np.power(NP_adj_z, 3) -3*NP_adj_z)*k3/6.
NP_corrected = NP_adj_Gauss + NP_third_order
NP_corrected /= np.sum(NP_corrected)
NP_Theory = NP_corrected - Gauss


plt.clf()
plt.scatter(x_grid, numerical, color='black', s=15)
plt.plot(x_grid, Theory, color='gray', linewidth = 2)
plt.plot(x_grid, NP_Theory, color='gray', linestyle='--', linewidth = 2)
plt.xlim(0.8*x_grid[0], 0.8*x_grid[-1])
plt.xlabel(r'$x$', size=14)
plt.ylabel(r'$\delta p$', size=14, rotation = 0)
plt.legend(['Numerical', 'Theory', r'$f_t = 0$ case'])
plt.tight_layout()
plt.savefig('numerical_path_impact.pdf')

np.save('NP_Theory.npy', NP_Theory)

plt.clf()
plt.scatter(x_grid, numerical-NP_Theory, color='black', s=15)
plt.plot(x_grid, Theory-NP_Theory, color='gray', linewidth = 2)
plt.xlim(0.8*x_grid[0], 0.8*x_grid[-1])
plt.xlabel('$x$', size=14)
plt.ylabel(r'$\widetilde{\delta p}$', size=14, rotation = 0)
plt.legend(['Numerical', 'Theory'])
plt.tight_layout()
plt.savefig('numerical_path_only.pdf')



