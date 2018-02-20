from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import cPickle
import os
from scipy.io import loadmat

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def layer_computation(x, W, b, act_func=lambda x: x):
    """ x is a vector in size of Nx1, W is a matrix in size NxL, where L is the 
    number of units in output layer, b is a vector with size Lx1"""
    L1 = np.dot(W.T, x) + b
    output = softmax(L1)
    return L1, output

def batch_layer_computation(X, W, b):
    """ X is an input matrix with size NxM, N is the number of input units, M is
    the number of training cases. W is a weight matrix NxL, L is the number of 
    output layers. Notice we need extra one row for bias, so we should append 
    another row vector [1, 1, 1, 1, ... ,1] to X to do implement the function. """
    one_array = np.ones(X.shape[1])
    X = np.vstack((X, one_array))
    print b.shape
    W = np.vstack((W, b))
    L1 = np.dot(W.T, X)
    Y = softmax(L1)
    return Y


def part3_cross_entropy(predicts, targets):
    """predicts and targets are matrices of size LxM, M is the nubmer training
    cases, L is the number of output units. we should transpose the matrix 
    before we pass the argument""" 
    total = -np.sum(np.multiply(np.log(predicts), targets))
    return np.true_divide(total, predicts.shape[1])

def CE_dWeight(X, Y, T):
    """X is the matrix of training sets with size NxM, N is the number of input
    units, M is the number of training cases, Y and T both has size LxM, L is the
    number of output layers, to make computation convinient, I will use this
    calculate the derivative of bias as well, so would append a row vector ones to
    X"""
    return np.dot(X, (Y-T).T)

def CE_dBias(Y, T):
    ones_arr = np.ones(Y.shape[1])
    return np.dot(ones_arr, (Y-T).T)

def approx_CE_dWeight_single(X, W, T, p, q, t):
    """Approximate the derivative of single entry in Jacobian matrix"""
    new_weight = W.copy()
    new_weight[p][q] += t
    b = np.zeros(W.shape[1])
    Y = batch_layer_computation(X, W, b)
    Y_h = batch_layer_computation(X, new_weight, b)
    df = part3_cross_entropy(Y_h, T) - part3_cross_entropy(Y, T)
    df = np.true_divide(df, t)
    return df



def approx_CE_dWeight(X, W, T, t=1e-4):
    size = W.shape
    df_matrix = np.empty(size, dtype=np.float64)
    for i in range(size[0]):
        for j in range(size[1]):
            df_matrix[i][j] = approx_CE_dWeight_single(X, W, T, i, j, t)
    return df_matrix

def initialize_weights(size, epsilon=0.4):
    """ Randomly initialize a matrix """
    W = np.random.rand(size[0], size[1])
    W = W * 2 * epsilon - epsilon
    return W

def initialize_bias(size):
    b = np.zeros(size)
    return b

def calculate_accuracy(Y, T):
    """ Calculate the performance , Y and T a matrices in size LxM """
    Y_cp = Y.copy()
    correct = 0
    total = Y.shape[1]
    for i in range(Y.shape[1]):
        ind_Y = argmax(Y[:, i])
        ind_T = argmax(T[:, i])
        if ind_Y == ind_T:
            correct += 1
    return np.true_divide(correct, total)

        

def initialize_targets(dataset, set_name, index):
    """ Initialize the target as a matrix, size is LxM, M is coincident with the
    matrix of the set_name in dictionary and index informs which index of that array 
    should be 1 """
    X = dataset[set_name]
    M = X.shape[0]
    T_index = np.zeros((M, 10))
    T_index[:, index] = 1
    return np.true_divide(X.T, 255.0), T_index.T

def seperate_train_valid():
    M = loadmat("mnist_all.mat")
    X_test = np.empty((784, 0), dtype=np.float)
    T_test = np.empty((10, 0), dtype=np.int)
    X_valid = np.empty((784, 0), dtype=np.float)
    T_valid = np.empty((10, 0), dtype=np.int)
    for i in range(10):
        X_test_i, T_test_i = initialize_targets(M, 'test'+str(i), i)
        num_sets = X_test_i.shape[1]
        valid_num = int(num_sets*0.2)
        test_num = num_sets - valid_num
        X_test = np.hstack((X_test, X_test_i[:,:test_num]))
        T_test = np.hstack((T_test, T_test_i[:,:test_num]))
        X_valid = np.hstack((X_valid, X_test_i[:,test_num:]))
        T_valid = np.hstack((T_valid, T_test_i[:,test_num:]))
    return X_test, T_test, X_valid, T_valid

def calculate_data(Y, T, Y_test, T_test, Y_valid, T_valid):
    train_performance = calculate_accuracy(Y, T)
    test_performance = calculate_accuracy(Y_test, T_test)
    valid_performance = calculate_accuracy(Y_valid, T_valid)
    train_cost = part3_cross_entropy(Y, T)
    test_cost = part3_cross_entropy(Y_test, T_test)
    valid_cost = part3_cross_entropy(Y_valid, T_valid)
    return np.array([train_performance, test_performance, valid_performance, train_cost, test_cost, valid_cost])

def gradient_descent(X, T, alpha=0.00001, EPS = 1e-5, max_iter = 300, mu = 0):
    """ X is the matrix of inputs, NxM, N is the number of input units, M is the 
    number of training cases. T is the matrix of results, size: LxM, L is the 
    number of output units """
    # Initialize a weight matrix
    size = [X.shape[0], T.shape[0]]
    W = initialize_weights(size)
    b = initialize_bias(T.shape[0])
    previous_W = W - 10*EPS
    previous_b = b - 10*EPS
    count = 0
    perform_dict = dict()

    summary = np.empty((6, 0), dtype=float)

    X_test, T_test, X_valid, T_valid = seperate_train_valid()

    p = 0
    
    while norm(W - previous_W)+norm(b - previous_b) > EPS and count < max_iter:
        previous_W = W.copy()
        b = b.copy()
        Y = batch_layer_computation(X, W, b)
        Y_test = batch_layer_computation(X_test, W, b)
        Y_valid = batch_layer_computation(X_valid, W, b)
        data = calculate_data(Y, T, Y_test, T_test, Y_valid, T_valid)
        summary = np.hstack((summary, data[:,None]))
        p = mu*p + alpha*CE_dWeight(X, Y, T)
        W = W - p
        b = b - alpha*CE_dBias(Y, T)
        print "Iter: " , count
        print "Weight: " , W
        print "Cost: " , part3_cross_entropy(Y, T)
        if count % (max_iter/5) == 0:
            print "Accuracy on test set: ", calculate_accuracy(Y, T)
            perform_dict[count] = (calculate_accuracy(Y, T), previous_W, previous_b)
        count += 1
    if mu == 0:
        np.save('tmp/part4_summary_data'+str(alpha), summary)
    else:
        np.save('tmp/part4_summary_data_momentum'+str(alpha), summary)
    return W, b, perform_dict

def part6_func(X, W, b, T, w1, w2):
    size = W.shape
    W1 = np.zeros(size)
    W2 = np.zeros(size)
    W1[299][8] = 1
    W2[399][8] = 1
    print W1.shape
    print W2.shape
    return part3_cross_entropy(batch_layer_computation(W + w1*W1 + w2*W2, X, b), T)

# =================================== Part 1 ===================================
# M = loadmat("mnist_all.mat")
# train_key = 'test'
# for k in range(10):
#   train_set = M[train_key+str(k)]
#   a = np.random.randint(train_set.shape[0], size = 10)
#   f, axarr = plt.subplots(5, 2)
#   for i in range(5):
#       for j in range(2):
#           axarr[i, j].imshow(train_set[i*2+j].reshape((28,28)), cmap = cm.gray)
#           axarr[i, j].axis('off')
#   f.subplots_adjust(hspace=0.3)
#   f.savefig('report/'+train_key+str(k)+'.png')
# plt.show()


# ===================================  Part 2 ===================================
# See function layer_computation and batch_layer_computation


# ===================================  Part 3 ===================================
# See function above function

# ===================================  Part 4 ===================================
# TODO: read the data from training set
M = loadmat("mnist_all.mat")
# Initialize input X and Target matrix
X = np.empty((784, 0), dtype=np.float)
T = np.empty((10, 0), dtype=np.int)
for i in range(10):
    X_i, T_i = initialize_targets(M, 'train'+str(i), i)
    X = np.hstack((X, X_i))
    T = np.hstack((T, T_i))

# W, b, perform_dict = gradient_descent(X, T)
# ===================================  Part 5 ===================================
# W, b, perform_dict = gradient_descent(X, T, mu=0.9)

# ===================================  Part 6 ===================================

W = np.load('tmp/part5_weight_5e-5_moment.npy')
b = np.load('tmp/part5_bias_5e-5_moment.npy')
# W is in size NxL, we choose 9th column and 300th and 400th entry in that column
W[299][8] = 0.0
W[399][8] = 0.0
w1 = np.arange(0.0, 1.0, 0.01)
w2 = np.arange(0.0, 1.0, 0.01)
# part6_func(X, W, b, T, w1, w2)