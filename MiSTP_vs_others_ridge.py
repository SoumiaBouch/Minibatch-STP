'''
The following code enables to compare the performance of MiSTP with three other zero order methods
on the ridge regression problem. Those methods are: Random Stochastic Gradient Free method (RSGF), 
Zero Order stochastic variance reduced method  (ZO-SVRG), and Zero Order Coordinates Descent method (ZO-CD)
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import getopt
import ast
import math
from math import ceil
from random import sample


# the objective function for the ridge regression problem
def objfun(x):
    return 1/(2*n)*(np.linalg.norm(np.dot(A,x)-y)**2)+(1/(2*n))*(np.linalg.norm(x)**2)


# The approximation of the objective function
def approx_func(batch,x):
    summation = 0
    for i in batch:
        summation = summation + (np.dot(A[i,:],x)-y[i])**2
    return (1/(2*len(batch)))*summation+(1/(2*n))*(np.linalg.norm(x)**2)


# function to sample the search direction from the uniform distribution over the unit sphere
def sample_spherical():
    vec = np.random.normal(size= (d,))
    vec /= np.linalg.norm(vec)
    return vec


# function to compute the gradient using a minibatch of the data
def minibatch_gradient_estimate(x):
    mu = 0.0001
    mini_batch = sample(range(n),batch_size)
    s = sample_spherical()
    g = d*((approx_func(mini_batch, x+mu*s) - approx_func(mini_batch, x))/mu)*s
    return g


def run_MiSTP():

    f_MiSTP = [[] for i in range(n_run)]
    
    for r in range(n_run):
        #initialisation
        x = starting_points[:,r]
        
        f_MiSTP[r].append(objfun(x))
        
        for e in range(n_epochs):
            for k in range(ceil(n/batch_size)):
                mini_batch = sample(range(n),batch_size)
                s_k = sample_spherical()
                x_plus = x + alpha_MiSTP*s_k
                x_minus = x - alpha_MiSTP*s_k
                array = np.array([approx_func(mini_batch,x_minus), approx_func(mini_batch,x_plus), approx_func(mini_batch,x)])
                indice_argmin = np.argmin(array)
                if (indice_argmin==0):
                    x = x_minus
                if (indice_argmin==1):
                    x = x_plus
                else:
                    x = x
            
            f_MiSTP[r].append(objfun(x))
                
    return f_MiSTP


def run_RSGF():
    
    # hyperparameters for RGF method
    mu = 0.0001
    
    f_RSGF = [[] for i in range(n_run)]
    
    for r in range(n_run):
    
        #initialisation
        x = starting_points[:,r]
    
        f_RSGF[r].append(objfun(x))
        
        for e in range(n_epochs):
            for k in range(ceil(n/batch_size)):
                mini_batch = sample(range(n),batch_size)
                s = sample_spherical()
                #update
                x = x - alpha_RSGF*(approx_func(mini_batch, x+mu*s) - approx_func(mini_batch, x))/mu*s
            f_RSGF[r].append(objfun(x))

    
    return(f_RSGF)


def run_ZO_SVRG():
    
    # hyperparameters for ZO_SVRG method
    mu = 0.0001
    
    f_ZO_SVRG = [[] for i in range(n_run)]
    
    for r in range(n_run):
        #initialisation
        x_0 = starting_points[:,r]

        f_ZO_SVRG[r].append(objfun(x_0))
        
        x_start = x_0
        x_k = x_0
        for e in range(n_epochs):
            # compute ZO estimate
            s = sample_spherical()
            g = d*((objfun(x_start+mu*s) - objfun(x_start))/mu)*s
            
            
            for k in range(ceil(n/batch_size)):
                v = minibatch_gradient_estimate(x_k) - minibatch_gradient_estimate(x_start) + g
                # update
                x_k = x_k - alpha_ZO_SVRG*v
            
            x_start = x_k
            f_ZO_SVRG[r].append(objfun(x_k))
    
    return f_ZO_SVRG


def run_ZO_CD():
    
    mu = 0.0001
    
    f_coord = [[] for i in range(n_run)]
    
    for r in range(n_run):
        
        #initialisation
        x = starting_points[:,r]
    
        f_coord[r].append(objfun(x))
        
        for e in range(n_epochs):
            for k in range(ceil(n/batch_size)):
                mini_batch = sample(range(n),batch_size)
                # compute gradient estimation
                I = np.identity(d)
                g = 0
                for i in range(d):
                    g += (approx_func(mini_batch, x+mu*I[:,i]) - approx_func(mini_batch, x-mu*I[:,i]))/(2*mu)*I[:,i]
                #update
                x = x - alpha_ZO_CD * g
            f_coord[r].append(objfun(x))

    return f_coord


def main(argv):

    global n_epochs,n_run,batch_size,alpha_MiSTP,alpha_RSGF,alpha_ZO_SVRG,alpha_ZO_CD

    n_epochs = None
    n_run = None
    batch_size = None
    alpha_MiSTP = None
    alpha_RSGF = None
    alpha_ZO_SVRG = None
    alpha_ZO_CD = None

    try:
        opts, args = getopt.getopt(argv[1:], '', ["n_epochs=", "n_run=", "batch_size=", "alpha_MiSTP=", "alpha_RSGF=", "alpha_ZO_SVRG=", "alpha_ZO_CD=" ])
    except:
        print("Error")

    for opt, arg in opts:
        if opt in ['--n_epochs']:
            n_epochs = arg
        elif opt in ['--n_run']:
            n_run = arg
        elif opt in ['--batch_size']:
            batch_size = arg
        elif opt in ['--alpha_MiSTP']:
            alpha_MiSTP = arg  
        elif opt in ['--alpha_RSGF']:
            alpha_RSGF = arg 
        elif opt in ['--alpha_ZO_SVRG']:
            alpha_ZO_SVRG = arg 
        elif opt in ['--alpha_ZO_CD']:
            alpha_ZO_CD = arg
    
    n_epochs = int(n_epochs)
    n_run = int(n_run)
    batch_size = int(batch_size)
    alpha_MiSTP = float(alpha_MiSTP)
    alpha_RSGF = float(alpha_RSGF)
    alpha_ZO_SVRG = float(alpha_ZO_SVRG)
    alpha_ZO_CD = float(alpha_ZO_CD)


    # read data
    data = pd.read_csv('splice_scale.csv', sep=',', header=None)

    # Number of samples
    global n
    n = data.shape[0]

    # Number of parameters
    global d
    d = data.shape[1]-1

    # data A & y for computing the objective function values
    global A
    global y
    A = np.array(data.iloc[: , 0:d])
    y = np.array(data.iloc[: , -1])
    
    # same starting points for all methods at each run
    global starting_points
    starting_points = np.random.normal(size= (d,n_run))

    # run experiment for each method
    f_RSGF = run_RSGF()
    f_MiSTP = run_MiSTP()
    f_ZO_SVRG = run_ZO_SVRG()
    f_ZO_CD = run_ZO_CD()


    # plotting the results

    # generating values for x_axis

    # at each epoch, MiSTP performs 3*batch_size*ceil(n/batch_size) function evaluations
    list_MiSTP = list(range(0, n_epochs*3*batch_size*ceil(n/batch_size)+1, 3*batch_size*ceil(n/batch_size)))

    # at each epoch, RSGF performs 2*batch_size*ceil(n/batch_size) function evaluations
    list_RGF = list(range(0, n_epochs*2*batch_size*ceil(n/batch_size)+1, 2*batch_size*ceil(n/batch_size)))

    # at each epoch, ZO_SVRG performs 4*batch_size*ceil(n/batch_size) function evaluations
    list_SVRG = list(range(0, n_epochs*4*batch_size*ceil(n/batch_size)+1, 4*batch_size*ceil(n/batch_size)))

    # at each epoch, ZO_CD performs 2*batch_size*ceil(n/batch_size)*d function evaluations
    list_CD = list(range(0, n_epochs*2*batch_size*d*ceil(n/batch_size)+1, 2*batch_size*ceil(n/batch_size)*d))

    plt.figure(figsize=(8.0, 5.0))
    plt.plot(list_MiSTP, np.mean(f_MiSTP, axis=0), label=r'$\tau = $'+str(batch_size)+r'$, \alpha = $'+str(alpha_MiSTP)+' MiSTP')
    plt.plot(list_RGF, np.mean(f_RSGF, axis=0), label=r'$\tau = $'+str(batch_size)+r'$, \alpha = $'+str(alpha_RSGF)+' RSGF')
    plt.plot(list_SVRG, np.mean(f_ZO_SVRG, axis=0), label=r'$\tau = $'+str(batch_size)+r'$, \alpha = $'+str(alpha_ZO_SVRG)+' ZO-SVRG')
    plt.plot(list_CD, np.mean(f_ZO_CD, axis=0), label=r'$\tau = $'+str(batch_size)+r'$, \alpha = $'+str(alpha_ZO_CD)+' ZO-CD')

    plt.fill_between(list_MiSTP,np.mean(f_MiSTP, axis=0) - 0.5*np.std(f_MiSTP, axis=0), np.mean(f_MiSTP, axis=0) + 0.5*np.std(f_MiSTP, axis=0), alpha=0.2)
    plt.fill_between(list_RGF,np.mean(f_RSGF, axis=0) - 0.5*np.std(f_RSGF, axis=0), np.mean(f_RSGF, axis=0) + 0.5*np.std(f_RSGF, axis=0), alpha=0.2)
    plt.fill_between(list_SVRG,np.mean(f_ZO_SVRG, axis=0) - 0.5*np.std(f_ZO_SVRG, axis=0), np.mean(f_ZO_SVRG, axis=0) + 0.5*np.std(f_ZO_SVRG, axis=0), alpha=0.2)
    plt.fill_between(list_CD,np.mean(f_ZO_CD, axis=0) - 0.5*np.std(f_ZO_CD, axis=0), np.mean(f_ZO_CD, axis=0) + 0.5*np.std(f_ZO_CD, axis=0), alpha=0.2)

    plt.xlabel("# of function queries",fontsize = 18)
    plt.ylabel(r'$f(x)$',fontsize = 18)

    plt.rcParams['xtick.labelsize']=18
    plt.rcParams['ytick.labelsize']=18

    plt.xlim(0, n_epochs*2*batch_size*ceil(n/batch_size))

    plt.ticklabel_format(axis="x", style="sci", scilimits=(3,3))

    plt.title(r'$n = $'+str(n)+', '+r'$d = $'+str(d),fontsize = 18)

    plt.legend(fontsize=16)

    plt.grid(linestyle = '--')




if __name__ == "__main__":
    main(sys.argv)