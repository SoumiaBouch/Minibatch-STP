'''
The following code enables to visualize the performance of MiSTP in a multi-layer neural network.
'''



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import getopt
import ast

def load_data(path):
    def one_hot(y):
        table = np.zeros((y.shape[0], 10))
        for i in range(y.shape[0]):
            table[i][int(y[i][0])] = 1 
        return table

    def normalize(x): 
        x = x / 255
        return x 

    data = np.loadtxt('{}'.format(path), delimiter = ',')
    return normalize(data[:,1:]),one_hot(data[:,:1])


def objfun(y, ypred):
    """ the loss function is the categorical crossentropy """
    return np.mean(np.sum(-y * np.log(ypred+10**(-15)),axis=1))

class NeuralNetwork:
    def __init__(self, X, y, batch = 64, lr = 1,  epochs = 50):
        self.input = X 
        self.target = y
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.loss = []
        self.acc = []
        self.init_weights()
      
    def init_weights(self):

        self.W1 = np.random.randn(self.input.shape[1],256)
        self.W2 = np.random.randn(self.W1.shape[1],128)
        self.W3 = np.random.randn(self.W2.shape[1],self.target.shape[1])

        self.b1 = np.random.randn(self.W1.shape[1],)
        self.b2 = np.random.randn(self.W2.shape[1],)
        self.b3 = np.random.randn(self.W3.shape[1],)

    def ReLU(self, x):
        return np.maximum(0,x)

    def softmax(self, z):
        z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
        return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)
    
    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]
        
    def feedforward(self):
        assert self.x.shape[1] == self.W1.shape[0]
        self.z1 = self.x.dot(self.W1) + self.b1
        self.a1 = self.ReLU(self.z1)

        assert self.a1.shape[1] == self.W2.shape[0]
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.ReLU(self.z2)

        assert self.a2.shape[1] == self.W3.shape[0]
        self.z3 = self.a2.dot(self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        self.objfunc = objfun(self.y, self.a3)
        
    def feedforward_plus(self):
        assert self.x.shape[1] == self.W1_plus.shape[0]
        self.z1 = self.x.dot(self.W1_plus) + self.b1_plus
        self.a1 = self.ReLU(self.z1)

        assert self.a1.shape[1] == self.W2_plus.shape[0]
        self.z2 = self.a1.dot(self.W2_plus) + self.b2_plus
        self.a2 = self.ReLU(self.z2)

        assert self.a2.shape[1] == self.W3_plus.shape[0]
        self.z3 = self.a2.dot(self.W3_plus) + self.b3_plus
        self.a3 = self.softmax(self.z3)
        self.objfunc_plus = objfun(self.y, self.a3)
        
    def feedforward_minus(self):
        assert self.x.shape[1] == self.W1_minus.shape[0]
        self.z1 = self.x.dot(self.W1_minus) + self.b1_minus
        self.a1 = self.ReLU(self.z1)

        assert self.a1.shape[1] == self.W2_minus.shape[0]
        self.z2 = self.a1.dot(self.W2_minus) + self.b2_minus
        self.a2 = self.ReLU(self.z2)

        assert self.a2.shape[1] == self.W3_minus.shape[0]
        self.z3 = self.a2.dot(self.W3_minus) + self.b3_minus
        self.a3 = self.softmax(self.z3)
        self.objfunc_minus = objfun(self.y, self.a3)

        
    def backprop(self):
        
        s_W1 = np.transpose(np.random.multivariate_normal(np.array([0]*self.W1.shape[0]) , np.identity(self.W1.shape[0]), self.W1.shape[1]))
        self.W1_plus = self.W1 + self.lr*s_W1
        self.W1_minus = self.W1 - self.lr*s_W1
        
        s_W2 = np.transpose(np.random.multivariate_normal(np.array([0]*self.W2.shape[0]) , np.identity(self.W2.shape[0]), self.W2.shape[1]))
        self.W2_plus = self.W2 + self.lr*s_W2
        self.W2_minus = self.W2 - self.lr*s_W2
        
        s_W3 = np.transpose(np.random.multivariate_normal(np.array([0]*self.W3.shape[0]) , np.identity(self.W3.shape[0]), self.W3.shape[1]))
        self.W3_plus = self.W3 + self.lr*s_W3
        self.W3_minus = self.W3 - self.lr*s_W3
        
        
        s_b1 = np.random.multivariate_normal(np.array([0]*self.b1.shape[0]) , np.identity(self.b1.shape[0]))
        self.b1_plus = self.b1 + self.lr*s_b1
        self.b1_minus = self.b1 - self.lr*s_b1
        
        s_b2 = np.random.multivariate_normal(np.array([0]*self.b2.shape[0]) , np.identity(self.b2.shape[0]))
        self.b2_plus = self.b2 + self.lr*s_b2
        self.b2_minus = self.b2 - self.lr*s_b2
        
        s_b3 = np.random.multivariate_normal(np.array([0]*self.b3.shape[0]) , np.identity(self.b3.shape[0]))
        self.b3_plus = self.b3 + self.lr*s_b3
        self.b3_minus = self.b3 - self.lr*s_b3
        
        # updating the weights
        
        self.feedforward_plus()
        self.feedforward_minus()
        
        array = np.array([self.objfunc_minus, self.objfunc_plus, self.objfunc])
        indice_argmin = np.argmin(array)
        if (indice_argmin==0):
            self.W1 = self.W1_minus
            self.W2 = self.W2_minus
            self.W3 = self.W3_minus
            self.b1 = self.b1_minus
            self.b2 = self.b2_minus
            self.b3 = self.b3_minus
        if (indice_argmin==1):
            self.W1 = self.W1_plus
            self.W2 = self.W2_plus
            self.W3 = self.W3_plus
            self.b1 = self.b1_plus
            self.b2 = self.b2_plus
            self.b3 = self.b3_plus
            
        
    def train(self):

        for epoch in range(self.epochs):
            l = 0
            acc = 0
            self.shuffle()

            data_indexes = np.arange(0, self.input.shape[0])
            mini_batches = [ data_indexes[k:k+self.batch] for k in range(0, self.input.shape[0], self.batch)]
            while len(mini_batches[-1]) != self.batch:
                mini_batches[-1] = np.append(mini_batches[-1],np.random.randint(0,self.input.shape[0]))
            
            for batch in mini_batches:
                self.x = self.input[batch]
                self.y = self.target[batch]
                self.feedforward()
                self.backprop()
                l+= self.objfunc
                acc+= np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.batch
            
            self.loss.append(l/(len(mini_batches)))
            self.acc.append(acc*100/(len(mini_batches)))
            


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

    # load data
    X_train, y_train = load_data(dataset)

    
    # run the experiment
    results_loss = []
    results_acc = []
    for k in range(len(minibatch_sizes)):
        minibatch_loss = []
        minibatch_acc = []
        for r in range(int(n_run)):
            NN = NeuralNetwork(X_train, y_train, batch=minibatch_sizes[k], lr = stepsizes[k], epochs=int(n_epochs))
            NN.train()
            minibatch_loss.append(NN.loss)
            minibatch_acc.append(NN.acc)
        
        results_loss.append(minibatch_loss)
        results_acc.append(minibatch_acc)

    # plotting the results

    # plotting the accuracy
    plt.figure(figsize=(12.0, 7.0))
    for k in range(len(minibatch_sizes)):
        plt.plot(np.mean(results_acc[k], axis=0), label=r'$\tau = $'+str(minibatch_sizes[k])+", "+ r'$\alpha=$'+str(stepsizes[k]))
        plt.fill_between(range(int(n_epochs)),np.mean(results_acc[k], axis=0) - 0.5*np.std(results_acc[k], axis=0), np.mean(results_acc[k], axis=0) + 0.5*np.std(results_acc[k], axis=0), alpha=0.2)

    plt.grid(linestyle = '--')
    plt.legend(fontsize=19, loc='lower right')
    plt.xlabel("Epochs", fontsize = 20)
    plt.ylabel("Accuracy (%)", fontsize = 20)
    plt.rcParams['xtick.labelsize']=18
    plt.rcParams['ytick.labelsize']=18
    plt.xlim(0,int(n_epochs))

    # plotting the loss
    plt.figure(figsize=(12.0, 7.0))
    for k in range(len(minibatch_sizes)):
        plt.plot(np.mean(results_loss[k], axis=0), label=r'$\tau = $'+str(minibatch_sizes[k])+", "+ r'$\alpha=$'+str(stepsizes[k]))
        plt.fill_between(range(int(n_epochs)),np.mean(results_loss[k], axis=0) - 0.5*np.std(results_loss[k], axis=0), np.mean(results_loss[k], axis=0) + 0.5*np.std(results_loss[k], axis=0), alpha=0.2)

    plt.grid(linestyle = '--')
    plt.legend(fontsize=19)
    plt.xlabel("Epochs", fontsize = 20)
    plt.ylabel("Loss", fontsize = 20)
    plt.rcParams['xtick.labelsize']=18
    plt.rcParams['ytick.labelsize']=18
    plt.xlim(0,int(n_epochs))




if __name__ == "__main__":
    main(sys.argv)