# Minibatch Stochastic Three Points Method for Unconstrained Smooth Minimization

This is the official implementation of the numerical experiments presented in the paper [Minibatch Stochastic Three Points Method for Unconstrained Smooth Minimization]().

## Requirements:

We conducted the experiments using:
- python 3.8.5
- pandas 1.1.3
- numpy 1.19.2
    
## Datasets:
For the experiments on ridge regression and regularized logistic regression problems, we used datasets from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) which are: abalone, a1a, australian, and splice. For the experiments on neural networks, we used the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of handwritten digits. We converted all the datasets to .csv files to conduct the experiments. Those datasets are available in this repository under the names:

  - abalone_scale.csv
  - a1a.csv
  - australian_scale.csv
  - splice_scale.csv
  - mnist_train.csv

To reproduce the numerical results presented in the paper please follow the instructions listed in the Jupyter notebook 'MiSTP_Implementations.ipynb'
