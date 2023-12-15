'''
The following code enables to visualize the performance of MiSTP and SGD on the ridge regression problem.
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import getopt


# the objective function for the ridge regression problem
def objfun(x):
    return 1/(2*n)*(np.linalg.norm(np.dot(A,x)-y)**2)+(1/(2*n))*(np.linalg.norm(x)**2)


# The approximation of the objective function
def approx_func(batch,x):
    summation = 0
    for i in batch:
        summation = summation + (np.dot(A[i,:],x)-y[i])**2
    return (1/(2*len(batch)))*summation+(1/(2*n))*(np.linalg.norm(x)**2)


def run_experiment( n_iter, n_run, batch_size, alpha_MiSTP, alpha_SGD):
    
    
    # initialization
    starting_points = np.random.normal(size= (d,n_run))  # to use the same starting point for MiSTP & SGD
    
    # generating the minibatches to use the same minibatch for both methods
    batches = {}
    for i in range(n_run):
        batches.update({i:np.random.randint(0, n, size = (n_iter,batch_size))})
    
    ########################## run the MiSTP method ################################
    
    # the mean and the covariance of the normal distribution 
    mean = np.array([0]*d) 
    cov = np.identity(d)
    
    f_MiSTP = np.zeros((n_run, n_iter+1))     #to store the objective function values
    approx_MiSTP = np.zeros((n_run, n_iter))  #to store the approximation of the objective function values
    for r in range(n_run):
        #initialisation
        x = starting_points[:,r]
        f_MiSTP[r,0] = objfun(x)
        for k in range(n_iter):
            # uniformally choose a batch
            batch = batches[r][k]
 
            s_k = np.random.multivariate_normal(mean, cov)
            x_plus = x + alpha_MiSTP*s_k
            x_minus = x - alpha_MiSTP*s_k
            array = np.array([approx_func(batch,x_minus), approx_func(batch,x_plus), approx_func(batch,x)])
            indice_argmin = np.argmin(array)
            if (indice_argmin==0):
                x = x_minus
            if (indice_argmin==1):
                x = x_plus
            else:
                x = x
            f_MiSTP[r,k+1] = objfun(x)
            approx_MiSTP[r,k] = approx_func(batch,x)
        
        
    ########################## run the SGD method ################################
        
    f_SGD = np.zeros((n_run, n_iter+1))      #to store the objective function values
    approx_SGD = np.zeros((n_run, n_iter))   #to store the approximation of the objective function values
    for r in range(n_run):
        x = starting_points[:,r]
        f_SGD[r,0] = objfun(x)
        for i in range(n_iter):
            # uniformally choose a batch
            batch = batches[r][i]
            # gradient computation
            summation = 0
            for j in batch:
                summation = summation + np.transpose(A[j,:])*(np.dot(A[j,:],x)-y[j])
            grad = (1/batch_size)*summation + (1/n)*x
            # SGD updates
            x = x - alpha_SGD * grad
            f_SGD[r,i+1] = objfun(x)
            approx_SGD[r,i] = approx_func(batch,x)
                
                
    ########################## plotting the results #######################
        
    # plotting the objective function
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(f_SGD.mean(0), label='SGD', color='teal')
    plt.plot(f_MiSTP.mean(0), label='MiSTP', color="blueviolet")
    plt.fill_between(range(n_iter+1),f_SGD.mean(0) - 0.5*np.std(f_SGD, axis=0), f_SGD.mean(0) + 0.5*np.std(f_SGD, axis=0), alpha=0.2, color='teal')
    plt.fill_between(range(n_iter+1),f_MiSTP.mean(0) - 0.5*np.std(f_MiSTP, axis=0), f_MiSTP.mean(0) + 0.5*np.std(f_MiSTP, axis=0), alpha=0.2, color="blueviolet")
    plt.xlim(0, n_iter)
    plt.ylim(0)
    plt.xlabel("Iterations",fontsize = 15)
    plt.ylabel(r'$f(x)$',fontsize = 15)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12
    plt.title(r'$\tau = %d$'%(batch_size)+", "+ r'$\alpha_{MiSTP} = $'+str(alpha_MiSTP)+", "+ r'$\alpha_{SGD} = $'+str(alpha_SGD),fontsize = 15)
    plt.legend(fontsize=14)
    plt.show()
        
    # plotting the approximation of the objective function
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(approx_SGD.mean(0), label='SGD', color='teal')
    plt.plot(approx_MiSTP.mean(0), label='MiSTP', color="blueviolet")
    plt.fill_between(range(n_iter),approx_SGD.mean(0) - 0.5*np.std(approx_SGD, axis=0), approx_SGD.mean(0) + 0.5*np.std(approx_SGD, axis=0), alpha=0.2, color='teal')
    plt.fill_between(range(n_iter),approx_MiSTP.mean(0) - 0.5*np.std(approx_MiSTP, axis=0), approx_MiSTP.mean(0) + 0.5*np.std(approx_MiSTP, axis=0), alpha=0.2, color="blueviolet")
    plt.xlim(0, n_iter)
    plt.ylim(0)
    plt.xlabel("Iterations",fontsize = 15)
    plt.ylabel(r'$\widetilde{f(}x)$',fontsize = 15)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12
    plt.title(r'$\tau = %d$'%(batch_size)+", "+ r'$\alpha_{MiSTP} = $'+str(alpha_MiSTP)+", "+ r'$\alpha_{SGD} = $'+str(alpha_SGD),fontsize = 15)
    plt.legend(fontsize=14)
    plt.show()




def run_experiment_full_batch( n_iter, n_run, alpha_STP, alpha_GD ):
    
    # initialization
    starting_points = np.random.normal(size= (d,n_run))  # to use the same starting point for MiSTP & SGD
    
    ######################## run the STP method ######################
    
    # the mean and the covariance of the normal distribution 
    mean = np.array([0]*d) 
    cov = np.identity(d)
    
    f_STP = np.zeros((n_run, n_iter+1)) #to store the objective function values
    for r in range(n_run):
        #initialisation
        x = starting_points[:,r]
        f_STP[r,0] = objfun(x)
        for k in range(n_iter):
            s_k = np.random.multivariate_normal(mean, cov)
            x_plus = x + alpha_STP*s_k
            x_minus = x - alpha_STP*s_k
            array = np.array([objfun(x_minus), objfun(x_plus), f_STP[r,k]])
            indice_argmin = np.argmin(array)
            if (indice_argmin==0):
                x = x_minus
            if (indice_argmin==1):
                x = x_plus
            else:
                x = x
            f_STP[r,k+1] = objfun(x)
            
            
    ######################## run the GD method ######################
    
    f_GD = np.zeros((n_run, n_iter+1))
    for r in range(n_run):
        x = starting_points[:,r]
        f_GD[r,0] = objfun(x)
        for i in range(n_iter):
            # gradient computation
            grad = (1/n)*np.dot(np.transpose(A),(np.dot(A,x)-y)) + (1/n)*x
            # SGD updates
            x = x - alpha_GD * grad
            f_GD[r,i+1] = objfun(x)

    
    ########################## plotting the results #######################
    
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(f_GD.mean(0), label='SGD', color='teal')
    plt.plot(f_STP.mean(0), label='MiSTP', color="blueviolet")
    plt.fill_between(range(n_iter+1),f_GD.mean(0) - 0.5*np.std(f_GD, axis=0), f_GD.mean(0) + 0.5*np.std(f_GD, axis=0), alpha=0.2, color='teal')
    plt.fill_between(range(n_iter+1),f_STP.mean(0) - 0.5*np.std(f_STP, axis=0), f_STP.mean(0) + 0.5*np.std(f_STP, axis=0), alpha=0.2, color="blueviolet")
    plt.xlim(0, n_iter)
    plt.ylim(0)
    plt.xlabel("Iterations",fontsize = 15)
    plt.ylabel(r'$f(x)$',fontsize = 15)
    plt.rcParams['xtick.labelsize']=12
    plt.rcParams['ytick.labelsize']=12
    plt.title(r'$\tau = n$'+", "+ r'$\alpha_{MiSTP} = $'+str(alpha_STP)+", "+ r'$\alpha_{SGD} = $'+str(alpha_GD),fontsize = 15)
    plt.legend(fontsize=14)
    plt.show()



def main(argv):

    dataset = ''
    n_iter = None
    n_run = None
    batch_size = None
    alpha_MiSTP = None
    alpha_SGD = None
    
    try:
        opts, args = getopt.getopt(argv[1:], '', ["dataset=", "n_iter=", "n_run=", "batch_size=", "alpha_MiSTP=", "alpha_SGD="])
    except:
        print("Error")

    for opt, arg in opts:
        if opt in ['--dataset']:
            dataset = arg
        elif opt in ['--n_iter']:
            n_iter = arg
        elif opt in ['--n_run']:
            n_run = arg
        elif opt in ['--batch_size']:
            batch_size = arg
        elif opt in ['--alpha_MiSTP']:
            alpha_MiSTP = arg
        elif opt in ['--alpha_SGD']:
            alpha_SGD = arg
    
    
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
    
    if (int(batch_size)==data.shape[0]): # if batch_size=full_batch (original STP & GD (gradient descent))
        run_experiment_full_batch( int(n_iter), int(n_run), float(alpha_MiSTP), float(alpha_SGD) )
    else:
        run_experiment( int(n_iter), int(n_run), int(batch_size), float(alpha_MiSTP), float(alpha_SGD))


if __name__ == "__main__":
    main(sys.argv)