'''
The following code enables to visualize the performance of MiSTP with different minibatch sizes on the regularized logistic regression problem.
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import getopt
import ast
import math


# the objective function for the regularized logistic regression problem
def objfun(x):
    summation = 0
    for i in range(n):
        summation = summation + math.log(1+math.exp(-y[i]*np.dot(A[i,:],x)))
    return 1/(2*n)*summation + (1/(2*n))*(np.linalg.norm(x)**2)



# The approximation of the objective function using a subset of the data
def approx_func(batch,x):
    summation = 0
    for i in batch:
        summation = summation + math.log(1+math.exp(-y[i]*np.dot(A[i,:],x)))
    return (1/(2*len(batch)))*summation+(1/(2*n))*(np.linalg.norm(x)**2)


def run_experiment(n_run, n_epoch, batch_size, alpha):
    
    # the mean and the covariance of the normal distribution for MiSTP
    mean = np.array([0]*d)
    cov = np.identity(d)

    f_MiSTP = np.zeros((n_run, n_epoch+1))   #to store the objective function values

    for r in range(n_run):
        #initialisation
        x = np.random.normal(size= (d,))
        f_MiSTP[r,0] = objfun(x)
        data_indexes = np.arange(0, n)
    
        for e in range(n_epoch):

            np.random.shuffle(data_indexes)

            # devide the dataset into subsets of size batch_size
            mini_batches = [ data_indexes[k:k+batch_size] for k in range(0, n, batch_size)]
            # if the last minibatch is not of size 'batch_size', complete it by randomly selecting samples from the dataset
            while len(mini_batches[-1]) != batch_size:   
                mini_batches[-1] = np.append(mini_batches[-1],np.random.randint(0,n))
            
            # run the MiSTP method
            for mini_batch in mini_batches:
                s_k = np.random.multivariate_normal(mean, cov)
                x_plus = x + alpha*s_k
                x_minus = x - alpha*s_k
                array = np.array([approx_func(mini_batch,x_minus), approx_func(mini_batch,x_plus), approx_func(mini_batch,x)])
                indice_argmin = np.argmin(array)
                if (indice_argmin==0):
                    x = x_minus
                if (indice_argmin==1):
                    x = x_plus
            f_MiSTP[r,e+1] = objfun(x)
    
    return f_MiSTP
    


def main(argv):

    dataset = ''
    n_epochs = None
    n_run = None
    minibatch_sizes = None
    stepsizes = None

    try:
        opts, args = getopt.getopt(argv[1:], '', ["dataset=", "n_epochs=", "n_run=", "minibatch_sizes=", "stepsizes=" ])
    except:
        print("Error")

    for opt, arg in opts:
        if opt in ['--dataset']:
            dataset = arg
        elif opt in ['--n_epochs']:
            n_epochs = arg
        elif opt in ['--n_run']:
            n_run = arg
        elif opt in ['--minibatch_sizes']:
            minibatch_sizes = arg
            minibatch_sizes = ast.literal_eval(minibatch_sizes)
        elif opt in ['--stepsizes']:
            stepsizes = arg
            stepsizes = ast.literal_eval(stepsizes)
    
    
    # read data
    data = pd.read_csv(dataset, sep=',', header=None)

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
    
    
    # run MiSTP for each minibatch size
    results = []
    for k in range(len(minibatch_sizes)):
        results.append(run_experiment(int(n_run), int(n_epochs), minibatch_sizes[k], stepsizes[k]))


    # plotting the results
    plt.figure(figsize=(8.0, 5.0))
    for i in range(len(results)):
        plt.plot(results[i].mean(0), label=r'$\tau = $'+str(minibatch_sizes[i])+r'$, \alpha = $'+str(stepsizes[i]))
        plt.fill_between(range(int(n_epochs)+1),results[i].mean(0) - 0.5*np.std(results[i], axis=0), results[i].mean(0) + 0.5*np.std(results[i], axis=0), alpha=0.2)
    
    plt.xlim(0,int(n_epochs))
    plt.xlabel("Epochs",fontsize = 18)
    plt.ylabel(r'$f(x)$',fontsize = 18)
    plt.rcParams['xtick.labelsize']=18
    plt.rcParams['ytick.labelsize']=18
    plt.title(r'$n = $'+str(n)+', '+r'$d = $'+str(d),fontsize = 18)
    plt.legend(fontsize=17)
    plt.grid(linestyle = '--')




if __name__ == "__main__":
    main(sys.argv)