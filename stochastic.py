import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.special import erf
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from numba import njit
import datetime
print(datetime.datetime.now())

# model parameter
sigma = 0.02

# numerical estimation parameters
n = 251 # must be ODD number for correctly placing the central node
integration_factor = 1.
statistical_factor = 2.
MC_samples = int(5e5)

# CPUs usage
fraction_cores = 1.

# utils
exclude_path = False
length_factor = 10.
rescale = 200
dtau_ML_factor = 2e3
n_std_grid = 6.
n_samples = 50
half_n_samples = n_samples/2
n_ML = n*rescale
T_ML = 10. #note alpha \sim r_0 = 1., SPDE timescale roughly 1. independent of sigma
corr_length = 1/sigma #note alpha \sim r_0 = 1.
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
factor_long = 100
dx_grid_long = dx_grid/factor_long
x_grid_long = np.array([(i+0.5-factor_long*n_samples/2)*dx_grid_long for i in range(factor_long*n_samples)])
split_sim = 10
T_MC= statistical_factor*1e7*(1/(split_sim*n_cores))
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
beta = sigma*np.sqrt(alpha)


Gauss = []
for x in x_grid_long:
    up = (1/2)*(1 + erf((x+dx_grid_long/2)/np.sqrt(2*nu)))
    down = (1/2)*(1 + erf((x-dx_grid_long/2)/np.sqrt(2*nu)))
    Gauss.append((up-down)/dx_grid_long)
Gauss = np.array(Gauss)
Gauss /= np.sum(Gauss)

small_Gauss = []
for i in range(n_samples):
    small_Gauss.append(np.sum(Gauss[i*factor_long:(i+1)*factor_long]))
small_Gauss = np.array(small_Gauss)


print(datetime.datetime.now())

plt.clf()
plt.plot(t_grid_ML, r_t, color = 'gray', linewidth = 0.8)
plt.plot(t_grid_ML, long_optimum, color = 'black', linewidth = 1.)
plt.axvline(t_grid_ML[mid_n_ML], color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.ylim(0.5*np.min(long_optimum), 1.35*np.max(long_optimum))
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


print('Path integrals...')

J = Parallel(n_jobs=n_cores)(
    delayed(compute_integral_J_montecarlo)(t) for t in t_from_point)
pre_K1 = Parallel(n_jobs=n_cores)(
    delayed(compute_integral_K1_montecarlo)(t, tp) for t, tp in t_pairs)
K1 = np.array(pre_K1).reshape((n, n))
pre_K2 = Parallel(n_jobs=n_cores)(
    delayed(compute_integral_K2_montecarlo)(t, tp) for t, tp in t_pairs)
K2 = np.array(pre_K2).reshape((n, n))


f = r_star - alpha
path_integral_J = np.sum(f*J)*dt
double_path_integral_K1 = np.sum(K1 * np.outer(f, f)) * dt**2
double_path_integral_K2 = np.sum(K2 * np.outer(f, f)) * dt**2


G1u = nu*np.exp(-sigma*np.sqrt(alpha)*np.abs(t_from_point))


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

print(path_integral_J/np.power(nu, 2))
print(double_path_integral_K1/np.power(nu, 2))
print(double_path_integral_K2/np.power(nu, 2))
print(spatial_var_correction/np.power(nu, 2))
print(path_var_excess/np.power(nu, 2))


plt.clf()
plt.plot(t_grid, f*J, color='black', linewidth = 1.5)
plt.xlabel('$t$', size=14)
plt.ylabel(r'$J_t \, f_t$', size = 14, rotation = 0, labelpad = 15)
plt.axvline(t_grid[mid_n], color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.axhline(0., color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.xlim(t_grid[0], t_grid[-1])
plt.tight_layout()
plt.savefig('J_integral.pdf')

plt.clf()
plt.plot(t_grid, G_f*J, color='black', linewidth = 1.5)
plt.xlabel('$t$', size=14)
plt.ylabel(r'$J_t \, \overline{f_{t}}$', size = 14, rotation = 0, labelpad = 15)
plt.axvline(t_grid[mid_n], color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.axhline(0., color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.xlim(t_grid[0], t_grid[-1])
plt.tight_layout()
plt.savefig('var_J_correction.pdf')

plt.clf()
plt.plot(t_grid, overline_f0_delta_Gf_f*J, color='black', linewidth = 1.5)
plt.xlabel('$t$', size=14)
plt.ylabel(r'$J_{t} \, \overline{f_0 \left(\overline{f_{t}} -\frac{1}{\alpha}f_{t} \right)}$', size = 14, labelpad = 15)
plt.axvline(t_grid[mid_n], color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.axhline(0., color = 'gray', linestyle = 'dashed', linewidth = 0.5)
plt.xlim(t_grid[0], t_grid[-1])
plt.tight_layout()
plt.savefig('var_excess.pdf')

plt.clf()
plt.imshow(K1 * np.outer(f, f), extent=[t_grid[0], t_grid[-1], t_grid[0], t_grid[-1]], cmap='Greys', origin='lower', interpolation='None')
plt.xlabel('$t$', size=14)
plt.ylabel("$t'$", size=14, rotation = 0)
plt.tight_layout()
plt.savefig('double_K1.pdf')

plt.clf()
plt.imshow(K2 * np.outer(f, f), extent=[t_grid[0], t_grid[-1], t_grid[0], t_grid[-1]], cmap='Greys', origin='lower', interpolation='None')
plt.xlabel('$t$', size=14)
plt.ylabel("$t'$", size=14, rotation = 0)
plt.tight_layout()
plt.savefig('double_K2.pdf')

plt.clf()
plt.imshow((K1+K2)*G_ff, extent=[t_grid[0], t_grid[-1], t_grid[0], t_grid[-1]], cmap='Greys', origin='lower', interpolation='None')
plt.xlabel('$t$', size=14)
plt.ylabel("$t'$", size=14, rotation = 0)
plt.tight_layout()
plt.savefig('double_K_G_ff.pdf')



np.save('t_grid.npy', t_grid)
np.save('f.npy', f)
np.save('J.npy', J)
np.save('K1.npy', K1)
np.save('K2.npy', K2)


z = x_grid/sqrt_nu
delta_nu = np.power(nu, 2)/9 + path_delta_nu
adj_nu = nu+delta_nu
delta_mu = -nu/2 + x_correction

adj_sqrt_nu = np.sqrt(adj_nu)

adj_Gauss = []
for x in x_grid_long:
    up = (1/2)*(1 + erf((x+dx_grid_long/2-delta_mu)/np.sqrt(2*adj_nu)))
    down = (1/2)*(1 + erf((x-dx_grid_long/2-delta_mu)/np.sqrt(2*adj_nu)))
    adj_Gauss.append((up-down)/dx_grid_long)
adj_Gauss = np.array(adj_Gauss)
adj_Gauss /= np.sum(adj_Gauss)

adj_z = (x_grid_long-delta_mu)/np.sqrt(adj_nu)
k3 = -np.sqrt(nu)/3
third_order = adj_Gauss*(np.power(adj_z, 3) -3*adj_z)*k3/6.
corrected = adj_Gauss + third_order
corrected /= np.sum(corrected)
Theory = corrected - Gauss

NP_delta_nu = np.power(nu, 2)/9
NP_adj_nu = nu+NP_delta_nu
NP_delta_mu = -nu/2
NP_adj_z = (x_grid_long-NP_delta_mu)/np.sqrt(NP_adj_nu)

NP_adj_Gauss = []
for x in x_grid_long:
    up = (1/2)*(1 + erf((x+dx_grid_long/2-NP_delta_mu)/np.sqrt(2*NP_adj_nu)))
    down = (1/2)*(1 + erf((x-dx_grid_long/2-NP_delta_mu)/np.sqrt(2*NP_adj_nu)))
    NP_adj_Gauss.append((up-down)/dx_grid_long)
NP_adj_Gauss = np.array(NP_adj_Gauss)
NP_adj_Gauss /= np.sum(NP_adj_Gauss)

NP_third_order = NP_adj_Gauss*(np.power(NP_adj_z, 3) -3*NP_adj_z)*k3/6.
NP_corrected = NP_adj_Gauss + NP_third_order
NP_corrected /= np.sum(NP_corrected)
NP_Theory = NP_corrected - Gauss

plt.clf()
plt.plot(x_grid_long, Theory/dx_grid_long, color='gray', linewidth = 2)
plt.plot(x_grid_long, NP_Theory/dx_grid_long, color='gray', linestyle='--', linewidth = 2)
plt.xlim(0.8*x_grid[0], 0.8*x_grid[-1])
plt.xlabel(r'$x$', size=14)
plt.ylabel(r'$\delta p$', size=14, rotation = 0, labelpad = 15)
plt.legend(['Theory', r'$f_t = 0$ case'])
plt.tight_layout()
plt.savefig('PREVIEW_numerical_path_impact.pdf')

print('Preview path impact !')

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



print('statistics...')
replicas = Parallel(n_jobs=n_cores)(delayed(Langevin_statistics)() for _ in tqdm(range(split_sim*n_cores)))
statistics = np.sum(replicas, axis = 0)
statistics /= np.sum(statistics)
numerical = statistics - small_Gauss
print(datetime.datetime.now())


np.save('x_grid.npy', x_grid)
np.save('numerical.npy', numerical)
np.save('theory.npy', Theory)
np.save('adj_Gauss.npy', adj_Gauss)
np.save('Gauss.npy', Gauss)


num_nu = np.sum(Gauss*np.power(x_grid_long, 2))
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


plt.clf()
plt.scatter(x_grid, numerical/dx_grid, color='black', s=20)
plt.plot(x_grid_long, Theory/dx_grid_long, color='gray', linewidth = 2)
plt.plot(x_grid_long, NP_Theory/dx_grid_long, color='gray', linestyle='--', linewidth = 2)
plt.xlim(0.8*x_grid[0], 0.8*x_grid[-1])
plt.xlabel(r'$x$', size=14)
plt.ylabel(r'$\delta p$', size=14, rotation = 0, labelpad = 15)
plt.legend(['Numerical', 'Theory', r'$f_t = 0$ case'])
plt.tight_layout()
plt.savefig('numerical_path_impact.pdf')

np.save('NP_Theory.npy', NP_Theory)

small_NP_Theory = []
for i in range(n_samples):
    small_NP_Theory.append(np.sum(NP_Theory[i*factor_long:(i+1)*factor_long]))
small_NP_Theory = np.array(small_NP_Theory)


plt.clf()
plt.scatter(x_grid, (numerical-small_NP_Theory)/dx_grid, color='black', s=20)
plt.plot(x_grid_long, (Theory-NP_Theory)/dx_grid_long, color='gray', linewidth = 2)
plt.xlim(0.8*x_grid[0], 0.8*x_grid[-1])
plt.xlabel('$x$', size=14)
plt.ylabel(r'$\delta p-\widetilde{\delta p}$', size=14, rotation = 0, labelpad = 15)
plt.legend(['Numerical', 'Theory'])
plt.tight_layout()
plt.savefig('numerical_path_only.pdf')


