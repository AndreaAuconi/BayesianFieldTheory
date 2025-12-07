from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import choice
from scipy.sparse import spdiags
import itertools
from scipy.sparse import csc_matrix, diags
import seaborn as sns
import ast
from numba import njit
import os
#os.chdir("/home/andrea/Desktop/Epidemie/")
import datetime
print(datetime.datetime.now())

fraction_cores = 0.98

# preprocessing parameters
only_Patricians = False
print('Only Patricians = ', only_Patricians)
load_names_dict = True
cut = 1379.
citations_max_gap = 15 # out if you are cited only once in more than _ years
min_gap = 0.2 # min inter-citations time
min_cits = 2 # fixed by model choice # DO NOT CHANGE
#set_similarity_threshold = 0.4
#strings_similarity_threshold = 0.25

# lambda estimation parameters
init_Year = 1334.5 # for the lambda estimation at border
upper_citations = 50 # +2, (i,f) #(the most cited are not counted for the lambda estimation)
lambda_Variance = 0.2 # uncertainty on lambda
n_samples = 100 # for the lambda exp distr factor

time_window = 7. #period pestem in deaths fraction estimation

# numerical parameters
n = 320
T_MC = 2000
T_MC_init = 0.05*T_MC
dtau = 0.0025
n_replicas = 200
sampling = int(1e2)
max_r = 0.2
T_init = 5.

# T_0 prior (uninformative)
B_init = -6.
A_init = 6.

n_cores = int(multiprocessing.cpu_count()*fraction_cores)


# %%

column_names = [
    'ID',
    'Type',
    'Name',
    't_m',
    't_i',
    't_f',
    'nMentions',
    'TimeSpan'
]

data_types = {
    't_i': float,
    't_f': float,
    'nMentions': int,
    'TimeSpan': float,
}

filtered = pd.read_csv('People_3dicFinal.csv', sep=';',
                       header=None, names=column_names, dtype=data_types,
                       decimal=',')
filtered = filtered[filtered['Type'] != 'x']

if only_Patricians:
   filtered = filtered[filtered['Type'] == 'p'] 


filtered['t_m'] = filtered['t_m'].apply(ast.literal_eval)

hm = 1/24
T_i = min(filtered['t_i'])-hm
T_f = max(filtered['t_f'])+hm

dt = (T_f - T_i)/n
t_grid = np.array([T_i + dt*(i+0.5) for i in range(n)])

noise_factor = np.sqrt(2*dtau/dt)

# %%

#r_c = np.median((filtered['nMentions']-1)/filtered['TimeSpan'])

int_init_Year = int((init_Year - T_i)/dt +0.5)

def get_index (t):
    return int((t-T_i)/dt)


raw = list(filtered['t_f'])
raw = np.array([get_index (t) for t in raw])
Deaths = [np.sum(raw==i) for i in range(n)]


raw = list(filtered['t_i'])
raw = np.array([get_index (t) for t in raw])
Born = [np.sum(raw==i) for i in range(n)]

Population = np.cumsum(Born) - np.cumsum(Deaths)
pre_Population = np.roll(Population, 1)
pre_Population[0] = 0


out = []
raw = [] 
for i in filtered['t_m']:
    set_cits = set([get_index (t) for t in i[1:-1]])
    if len(set_cits) < upper_citations:
        for k in list(set_cits):
            raw.append(k)
    else:
        out.append([get_index(i[0]), get_index(i[-1])])
raw = np.array(raw)
lam_freq = np.array([float(np.sum(raw==i)) for i in range(n)])
for this in out:
    i, f = this
    vec = []
    w = upper_citations/(f-i-1)
    for j in range(n):
        if i<j and j<f:
            vec.append(w)
        else:
            vec.append(0.)
    lam_freq += np.array(vec)
   

lam_freq = lam_freq/((1+pre_Population)*dt)


avg_lam_freq = np.mean(lam_freq[int_init_Year:-int_init_Year])
for i in range(n):
    if i < int_init_Year or i > n-int_init_Year-1:
        lam_freq[i] = avg_lam_freq

plt.clf()
plt.plot(t_grid, lam_freq)
plt.ylabel('$\lambda$', size = 14, rotation = 0)
plt.xlabel('Year', size = 13)
plt.savefig('Lambda.pdf')

dr = max_r/n

left_border_vec = np.zeros_like(t_grid)
left_border_vec[0] = 1
right_border_vec = np.zeros_like(t_grid)
right_border_vec[-1] = 1

#Q = np.array([[np.prod(1-lam_freq[t1: t2+1]*dt) for t2 in range(n)] for t1 in range(n)])


print('Sampling Q ...')

def Q_sampling(n_samples):
    def get_Q (xi):
        return np.array([[np.prod(1-xi*lam_freq[t1: t2+1]*dt) for t2 in range(n)] for t1 in range(n)])
    Q = np.zeros(shape=(n, n))
    for _ in range(n_samples):
        xi = np.random.lognormal(-0.5*lambda_Variance, np.sqrt(lambda_Variance))
        Q += get_Q (xi)
    Q /= n_samples
    return Q

Q_samples = Parallel(n_jobs=n_cores)(delayed(Q_sampling)(n_samples) for _ in tqdm(range(100)))
Q = np.mean(Q_samples, axis = 0)

plt.clf()
plt.imshow(Q)
plt.savefig('Q.pdf')


# %%


print(datetime.datetime.now())

names = []
stories = []
t_mentions = [] 
for k in filtered.index:
    i = get_index(filtered['t_i'][k])
    f = get_index(filtered['t_f'][k])
    name = filtered['Name'][k]
    if f > i:
        stories.append([i, f])
        names.append(name)
        t_mentions.append(filtered['t_m'][k])

names = np.array(names)
stories = np.array(stories)


# %%


@njit
def dV_ds_data (r):
    S = np.empty((n, n), dtype=np.float64)
    for t1 in range(n):
        for t2 in range(n):
            S[t1, t2] = np.prod(1-r[t1: t2+1]*dt)
    A = S*Q
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n-1):
        this = A[i+1, n-1]
        G[i, n-1] = this
        f = n-2
        while f >= i:
            this += dt*r[f+1]*A[i+1, f]
            G[i, f] = this
            f -= 1
    def dG_dr (i, f):
        def comp (j):
            if i < j:
                if f < j:
                    term = A[i+1, j-1] - G[i, j]/(1-dt*r[j])
                else:
                    term = - G[i, f]/(1-dt*r[j])
                return dt * term 
            else:
                return 0.
        return np.array([comp (j) for j in range(n)])   
    vec = np.zeros((n), dtype=np.float64)
    for this in stories:
        i, f = this
        vec_up = - dG_dr (i, f) / G[i, f]
        vec_down = - dG_dr (i, i) / (1 - G[i, i])
        vec += vec_up + vec_down      
    return vec*r



print(datetime.datetime.now())

print('Saddle point approximation...')

# SADDLE POINT APPROX
param_sigma_prior = 0.#uninformative

n_sigma = 140
sigma_list = [0.01*(1.035**i) for i in range(n_sigma)]


@njit
def V_data (r):
    S = np.empty((n, n), dtype=np.float64)
    for t1 in range(n):
        for t2 in range(n):
            S[t1, t2] = np.prod(1-r[t1: t2+1]*dt)
    A = S*Q
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n-1):
        this = A[i+1, n-1]
        G[i, n-1] = this
        f = n-2
        while f >= i:
            this += dt*r[f+1]*A[i+1, f]
            G[i, f] = this
            f -= 1
    V = 0.
    for this in stories:
        i, f = this
        V += np.log((1-G[i,i])/G[i, f])
    return V


@njit
def Hessian_V_s_data (r):
    dt2 = np.power(dt, 2)
    S = np.empty((n, n), dtype=np.float64)
    for t1 in range(n):
        for t2 in range(n):
            S[t1, t2] = np.prod(1-r[t1: t2+1]*dt)
    A = S*Q
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n-1):
        this = A[i+1, n-1]
        G[i, n-1] = this
        f = n-2
        while f >= i:
            this += dt*r[f+1]*A[i+1, f]
            G[i, f] = this
            f -= 1
    def dG_dr (i, f):
        def comp (j):
            if i < j:
                if f < j:
                    term = A[i+1, j-1] - G[i, j]/(1-dt*r[j])
                else:
                    term = - G[i, f]/(1-dt*r[j])
                return dt * term 
            else:
                return 0.
        return np.array([comp (j) for j in range(n)])       
    def d2G_drdr (j, k, i, f):
        if i < min(j, k):
            if f < max(j, k):
                if k < j:
                    return dt2*(G[i, j]/(1-dt*r[j]) -A[i+1, j-1])/(1-dt*r[k]) 
                elif k > j:
                    return dt2*(G[i, k]/(1-dt*r[k]) -A[i+1, k-1])/(1-dt*r[j]) 
                else:
                    return 0.
            else:
                if k == j:
                    return 0.
                else:
                    return dt2*G[i, f]/((1-dt*r[j])*(1-dt*r[k]))
        else:
            return 0.
    H = np.zeros((n, n))
    for this in stories:
        i, f = this
        dG_dr_i_f = dG_dr (i, f)
        dG_dr_i_i = dG_dr (i, i)
        den_i = 1-G[i, i]
        den_i_2 = np.power(den_i, 2)
        den_f = G[i, f]
        den_f_2 = np.power(den_f, 2)
        for j in range(n):
            for k in range(n):
                up = - d2G_drdr (j, k, i, i) / den_i - dG_dr_i_i [j] * dG_dr_i_i [k] / den_i_2
                down = - d2G_drdr (j, k, i, f) / den_f + dG_dr_i_f [j] * dG_dr_i_f [k] / den_f_2
                H[j, k] += up + down
    for j in range(n):
        for k in range(n):
            H[j, k] *= r[j] * r[k]
            if k == j:
                H[j, k] += dV_ds_data(r)[j]               
    return H


@njit
def U_opt(r_opt, sigma_prime):
    s_opt = np.log(r_opt)
    Qvar = np.sum(np.power(np.roll(s_opt, -1) - s_opt, 2)[:-1])
    terms_prior = np.power(s_opt[0]-B_init, 2)/(2*A_init) + (s_opt[-1]-s_opt[0])/2 +Qvar/(2*np.power(sigma_prime, 2)*dt)
    terms_measurement = V_data (r_opt)
    return terms_prior + terms_measurement

def log_det_H_opt(r_opt, sigma_prime):
    m_diag = left_border_vec/A_init + 2/(np.power(sigma_prime, 2)*dt)
    up_down = -np.ones(shape = (len(m_diag)-1))/(np.power(sigma_prime, 2)*dt)
    H_prior = csc_matrix(diags([m_diag, up_down, up_down], [0, 1, -1])).toarray()
    H_data = Hessian_V_s_data(r_opt)
    H = H_data + H_prior 
    return np.prod(np.linalg.slogdet(H))

def ln_I (r_opt, sigma_prime):
    return -U_opt(r_opt, sigma_prime) + (n/2)*np.log(2*np.pi) -0.5*log_det_H_opt(r_opt, sigma_prime) 

def saddle_point (sigma_prime):
    sigma2 = sigma_prime**2
    main_diag = -2 * np.ones(n)
    main_diag[0] = -1
    main_diag[n-1] = -1
    off_diag = np.ones(n)
    L_lil = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], n, n, format='lil')
    L = L_lil.tocsc() / dt**2
    M = spdiags(np.ones(n), 0, n, n, format='csc') - (dtau / sigma2) * L
    a_Langevin = M.diagonal(k=-1)
    b_Langevin = M.diagonal(k=0)
    c_Langevin = M.diagonal(k=1)
    
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
    def find_optimum(sigma):
        x = np.ones_like(t_grid)*-3.
        tau = np.float64(0.)
        while tau < T_init:
            tau += dtau
            border_terms = -left_border_vec*(x-B_init)/A_init + (left_border_vec-right_border_vec)/2
            rhs = x + (dtau/dt) * (-dV_ds_data(np.exp(x)) + border_terms)
            x = thomas_algorithm_Langevin(rhs) 
        return x
    
    s_opt = find_optimum(sigma_prime)
    r_opt = np.exp(s_opt)
    approx = ln_I(r_opt, sigma_prime) -n*np.log(sigma_prime) -np.power(sigma_prime, 2)*(T_f-T_i)/8
    return s_opt, approx

# %%

pre_samples = Parallel(n_jobs=n_cores)(delayed(saddle_point)(sigma_list[i]) for i in tqdm(range(n_sigma)))
optima = np.array([i[0] for i in pre_samples])
integral_list = np.array([i[1] for i in pre_samples])

integral_list -= np.array(sigma_list)*param_sigma_prior
integral_list -= np.min(integral_list)

sigma_posterior = np.exp(integral_list)
sigma_posterior /= np.sum(sigma_posterior)

plt.clf()
plt.plot(sigma_list, sigma_posterior)
plt.xlim(0., max(sigma_list))
this_max = np.max(sigma_posterior)
plt.ylim(0., this_max*1.15)
plt.ylabel('$p(\sigma | D)$', size = 15, rotation = 0)
plt.xlabel('$\sigma$', size = 14)
plt.tight_layout()
plt.savefig('sigma_posterior.pdf')

plt.clf()
plt.plot(sigma_list, sigma_posterior)
plt.xlim(min(sigma_list), max(sigma_list))
this_max = np.max(sigma_posterior)
plt.ylim(this_max*1e-4, this_max*2)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('$p(\sigma | D)$', size = 15, rotation = 0)
plt.xlabel('$\sigma$', size = 14)
plt.tight_layout()
plt.savefig('log_sigma_posterior_log_scale.pdf')

# %%

m = max(sigma_posterior) 
ML_index = list(sigma_posterior).index(m)
s_ML = optima[ML_index]
peak = max(s_ML[:int(n/2)])#n/2 to avoid border peaks
t_BD = t_grid[list(s_ML).index(peak)]

t_A = get_index (t_BD - time_window/2)
t_B = get_index (t_BD + time_window/2)

print('t_BD = ' + str(t_BD))


# %%

def curve_sampling(x0, sigma):
    
    sigma2 = sigma**2
    main_diag = -2 * np.ones(n)
    main_diag[0] = -1
    main_diag[n-1] = -1
    off_diag = np.ones(n)
    L_lil = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], n, n, format='lil')
    L = L_lil.tocsc() / dt**2
    M = spdiags(np.ones(n), 0, n, n, format='csc') - (dtau / sigma2) * L
    a_Langevin = M.diagonal(k=-1)
    b_Langevin = M.diagonal(k=0)
    c_Langevin = M.diagonal(k=1)
    
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
    def curve_statistics():
        def noise():
            return np.random.normal(0, noise_factor, n)
        x = x0
        M = np.zeros(shape=(n, n))
        i = 0
        tau = np.float64(0.)
        samples = []
        while tau < T_MC:
            tau += dtau
            border_terms = -left_border_vec*(x-B_init)/A_init + (left_border_vec-right_border_vec)/2
            rhs = x + (dtau/dt) * (-dV_ds_data(np.exp(x)) + border_terms) + noise()
            x = thomas_algorithm_Langevin(rhs)           
            i += 1
            if i == sampling:
                if tau > T_MC_init:
                    r_vec = np.exp(x)
                    for index in range(n):
                        this_r = r_vec[index]
                        if this_r < max_r:
                            h = int(this_r/dr)
                            M[index, h] += 1
                    r_A, r_B = r_vec[t_A], r_vec[t_B]
                    r_sub = (r_A+r_B)/2.
                    excess_Deaths = np.sum((r_vec[t_A: t_B+1]-r_sub)*dt)
                    samples.append(excess_Deaths)
                i = 0
        return M, samples

    M, samples = curve_statistics()
    return M, samples

samples = [choice([i for i in range(n_sigma)], p = sigma_posterior) for _ in range(n_replicas)]

sigma_samples = [sigma_list[i] for i in samples]
optima_samples =  [optima[i] for i in samples]


print(datetime.datetime.now())
print('Sampling curve ...')
replicas = Parallel(n_jobs=n_cores)(delayed(curve_sampling)(optima_samples[this], sigma_samples[this]) for this in tqdm(range(n_replicas)))
print(datetime.datetime.now())

curve = []
pre_hist_data = [] 
for i in replicas:
    curve.append(i[0])
    pre_hist_data.append(i[1])
    
M = np.sum(curve, axis = 0)
M /= np.transpose(np.array([np.sum(M, axis=1)] * n))

plt.clf()
plt.figure(figsize=(7, 5))
plt.imshow(np.flip(np.transpose(M), axis=0), extent=[t_grid[0], t_grid[-1], 0, max_r], alpha = 1., cmap = "gray_r", aspect="auto")
plt.ylabel('$r$', size = 15, rotation = 0)
plt.xlabel('t', size = 14)
plt.xlim(t_grid[0], t_grid[-1])
plt.ylim(0, max_r)
plt.tight_layout()
plt.savefig('nonparametric_MCMC.pdf')

plt.clf()
plt.figure(figsize=(7, 5))
plt.imshow(np.flip(np.transpose(M), axis=0), extent=[t_grid[0], t_grid[-1], 0, max_r], alpha = 1., aspect="auto")
plt.ylabel('$r$', size = 15, rotation = 0)
plt.xlabel('t', size = 14)
plt.xlim(t_grid[0], t_grid[-1])
plt.ylim(0, max_r)
plt.tight_layout()
plt.savefig('nonparametric_MCMC_color.pdf')


hist_data = list(itertools.chain.from_iterable(pre_hist_data))

plt.clf()
sns.histplot(np.array(hist_data), kde=True)
plt.xlabel("Deaths fraction", size = 14)
plt.ylabel('Posterior', size = 14)
plt.savefig('Mortality.pdf')

plt.clf()
sns.ecdfplot(hist_data)
plt.xlabel("Deaths fraction", size = 14)
plt.ylabel('Cumulative posterior', size = 14)
plt.savefig('Cum_Mortality.pdf')


with open('M_curve.pkl', 'wb') as file:
    pickle.dump(M, file)
with open('hist_data.pkl', 'wb') as file:
    pickle.dump(hist_data, file)
np.savetxt("M_curve.txt", M, fmt="%.6f")
np.savetxt("hist_data.txt", np.array(hist_data), fmt="%.6f")


mean_value_total = np.mean(hist_data)
std_dev_total = np.std(hist_data)

output_filename = 'mortality_stats.txt'
output_text = (
    f"{mean_value_total:.6f}\n"
    f"{std_dev_total:.6f}"
)

with open(output_filename, 'w') as file:
        file.write(output_text)

print('Mean:')
print(mean_value_total)
print('Std:')
print(std_dev_total)

# %%

r_disc = np.array([dr*(i+0.5) for i in range(n)]) 

mean_r_t = []
std_r_t = []

for row in range(n):
    y = M[row]
    avg = np.sum(y*r_disc)
    var = np.sum(y*np.power(r_disc, 2))
    var -= np.power(avg, 2)
    mean_r_t.append(avg)
    std_r_t.append(np.sqrt(var))

mean_r_t = np.array(mean_r_t)
std_r_t = np.array(std_r_t)


with open('Mean.txt', 'w') as file:
    for this in mean_r_t:
        file.write(str(this))
        file.write('\n')

with open('Std.txt', 'w') as file:
    for this in std_r_t:
        file.write(str(this))
        file.write('\n')

with open('t_grid.txt', 'w') as file:
    for this in t_grid:
        file.write(str(this))
        file.write('\n')
        

# %%


print(datetime.datetime.now())

print('tau estimates ...')

isExist = os.path.exists('vite')
if not isExist:
    os.mkdir('vite')

@njit
def tau_estimate (i, f, r_ML):
    S = np.empty((n, n), dtype=np.float64)
    for t1 in range(n):
        for t2 in range(n):
            S[t1, t2] = np.prod(1-r_ML[t1: t2+1]*dt)
    A = S*Q
    out = A[f+1, n-1]
    def term (tau):
        if tau <= f:
            return 0.
        else:
            return dt*r_ML[tau]*A[f+1, tau-1]
    terms = np.array([term (tau) for tau in range(n)])
    Z = np.sum(terms) + out
    def surv (tau):
        if tau <= f:
            return 0.
        else:
            return 1-np.sum(terms[:tau])/Z
    return np.array([surv (tau) for tau in range(n)])
  

for _ in range(50):
    k = int(np.random.uniform(0, len(names)))  
    name = names[k]
    i, f = stories[k]
    mentions = t_mentions[k] 
    vec_i_f = np.array([1.*(j>=i)*(j<=f) for j in range(n)]) 
    profile = vec_i_f + tau_estimate (i, f, mean_r_t)
    plt.clf()
    plt.figure(figsize=(8, 3))
    plt.plot(t_grid, profile, color = 'Black')
    for t in mentions:
        plt.vlines(t_grid[get_index(t)], 0, 1, color ='gray')
    plt.title(name)
    plt.savefig('vite/' + name +'.pdf')
    plt.close()
    with open('vite/' + name + '.txt', 'w') as file:
        for this in profile:
            file.write(str(this))
            file.write('\n')



pre_delibere = pickle.load(open("Data.pkl", "rb"))
pre_delibere.sort(key=lambda elem: elem[0])

delibere = []
for i in pre_delibere:
    if i[0] < cut:
        delibere.append(i)
        
list_mortalitatis_mentions = []
examples_pestem = []

for i in range(len(delibere)):
    words = delibere[i][2].split()
    Id = 0
    while Id < len(words)-2:
        word = words[Id]
        if word.lower() in ['epydimie', 'epidimia', 'epydemiam', 'pestem', 'mortalitatis']:
            list_mortalitatis_mentions.append(delibere[i][0])
            examples_pestem.append(delibere[i][2])
        Id += 1



max_plot = 1.25*np.max(mean_r_t+std_r_t)

plt.clf()
plt.figure(figsize=(8, 3))
hist_counts, bin_edges = np.histogram(list_mortalitatis_mentions, bins=50)
plt.clf()
plt.plot(t_grid, mean_r_t, label='Mean mortality rate')
plt.plot(t_grid, mean_r_t+std_r_t, color = 'gray', linestyle = 'dashed', label='$\pm 1$ Std')
plt.plot(t_grid, mean_r_t-std_r_t, color = 'gray', linestyle = 'dashed')
#plt.plot(t_grid, np.exp(s_ML), label=r'$\exp (s_{ML})$')
bin_widths = np.diff(bin_edges)
bin_centers = bin_edges[:-1] + bin_widths/2
height_scaling_factor = 0.01  # Adjust this as needed
plt.bar(bin_centers, height_scaling_factor*hist_counts, color = 'red',
        width=0.8*bin_widths, alpha=0.7, label='Mentions of "pestem"')
plt.legend()
plt.ylabel('$r$', size=14, rotation=0)
plt.xlabel('Year', size=13)
plt.ylim(0, max_plot)
plt.xlim(t_grid[0], t_grid[-1])
plt.tight_layout()
plt.savefig('Mean_curve_1.pdf')


plt.clf()
plt.figure(figsize=(8, 3))
hist_counts, bin_edges = np.histogram(list_mortalitatis_mentions, bins=50)
plt.clf()
plt.plot(t_grid, mean_r_t, color = "purple", label='Mean mortality rate')
plt.plot(t_grid, mean_r_t+std_r_t, color = 'gray', linestyle = 'dashed', label='$\pm 1$ Std')
plt.plot(t_grid, mean_r_t-std_r_t, color = 'gray', linestyle = 'dashed')
#plt.plot(t_grid, np.exp(s_ML), label=r'$\exp (s_{ML})$')
plt.imshow(np.flip(np.transpose(M), axis=0), extent=[t_grid[0], t_grid[-1], 0, max_r], alpha = 0.75, cmap = "gray_r", aspect="auto")
bin_widths = np.diff(bin_edges)
bin_centers = bin_edges[:-1] + bin_widths/2
height_scaling_factor = 0.01  # Adjust this as needed
plt.bar(bin_centers, height_scaling_factor*hist_counts, color = 'red',       width=0.8*bin_widths, alpha=0.7, label='Mentions of "pestem"')
plt.legend()
plt.ylabel('$r$', size=14, rotation=0)
plt.xlabel('Year', size=13)
plt.ylim(0, max_plot)
plt.xlim(t_grid[0], t_grid[-1])
plt.tight_layout()
plt.savefig('Mean_curve2.pdf')

