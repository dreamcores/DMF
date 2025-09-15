import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fftpack import dct, idct
import kwant
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  

from disorder_system import *

def generate_correlated_disorder(Num_sites, Mean, correlation_length, Std_deviation, N):

    disorder = np.random.normal(Mean, Std_deviation, Num_sites)
    disorder -= np.mean(disorder)
    
    correlated_disorder = gaussian_filter(disorder, sigma=correlation_length)
    dct_disorder = dct(correlated_disorder, norm='ortho')

    ck = np.zeros_like(dct_disorder)
    ck[:N + 1] = dct_disorder[:N + 1]

    Y = ck[1:N + 1]
    return Y


def generate_disorder(Num_sites, Mean, Std_deviation, N):

    disorder = np.random.normal(Mean, Std_deviation, Num_sites)
    disorder -= np.mean(disorder)
    dct_disorder = dct(disorder, norm='ortho')
    ck = np.zeros_like(dct_disorder)
    ck[:N + 1] = dct_disorder[:N + 1]

    Y = ck[1:N + 1]
    return Y

def disorder_pot(Y, Num_sites):
    X = np.zeros(Num_sites)
    X[1:1 + len(Y)] = Y  
    return idct(X, norm='ortho')



'''data dim'''
bias = [-0.04, -0.02, 0, 0.02, 0.04]
Vw = np.linspace(0, 0.3, 15)
mu_w = np.linspace(0, 0.2, 15)

num_sequences = 10000  # Number of Y sequences to generate



Y0 = []
Y1 = []

# Loop to generate multiple sequences
for num in range(num_sequences):
    Y_chem = generate_correlated_disorder(Num_sites = 301,
                                          Mean = 0,
                                          correlation_length = 3,
                                          Std_deviation = 0.5,       # chem deviation
                                          N = 5)
    
    Y_pair = generate_disorder(Num_sites = 301,
                                          Mean = 0,
                                          Std_deviation = 0.05,       # Pair deviation
                                          N = 5)


    Y0.append(Y_chem)
    Y1.append(Y_pair)

Y = np.stack([Y0, Y1], axis=1)


def compute_cond_fine(args):

    Y, bias, Vw, mu_w = args 

    Vdis = disorder_pot(Y[0], Num_sites = 301)

    Pairdis = disorder_pot(Y[1], Num_sites = 301)
    
    sys = BdG_conds_1D(Vw, mu_w, Vdis, Pairdis).finalized()
    smatrix = kwant.smatrix(sys, energy=bias, check_hermiticity=True)
    
    Cond_LL = local_cond(np.array(smatrix.submatrix(0, 0)))
    Cond_RR = local_cond(np.array(smatrix.submatrix(1, 1)))
    Cond_LR = nonlocal_cond(np.array(smatrix.submatrix(0, 1)))
    Cond_RL = nonlocal_cond(np.array(smatrix.submatrix(1, 0)))

    return  Cond_LL, Cond_RR, Cond_LR, Cond_RL



if __name__ == "__main__":

    # Prepare task arguments
    task_args = [
        (Y[num], bias[i], Vw[j], mu_w[k])
        for num in range(len(Y))
        for i in range(len(bias))
        for j in range(len(Vw))
        for k in range(len(mu_w))
    ]
    
    total_tasks = len(task_args)
    print(f"Total number of tasks: {total_tasks}")
    
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    t1 = time.time()
    
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(compute_cond_fine, task_args), total=total_tasks))
    
    t2 = time.time()
    print(f"Total computation time: {(t2 - t1)/3600:.5f} hours")
    
    # Reshape results
    results_array = np.array(results)
    results_shaped = results_array.reshape(len(Y), len(bias), len(Vw), len(mu_w), 4)
    Y = np.array(Y)
    
    print(results_shaped.shape)
    print(Y.shape)
    
    # Save results
    np.save('datasets.npy', results_shaped)
    np.save('labels.npy', Y)
    
