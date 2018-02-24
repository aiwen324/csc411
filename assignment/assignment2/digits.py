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
    summary = np.empty((6, 0), dtype=float)
    X_test, T_test, X_valid, T_valid = seperate_train_valid()
    p = 0
    while norm(W - previous_W)+norm(b - previous_b) > EPS and count < max_iter:
        previous_W = W.copy()
        previous_b = b.copy()
        # Getting data for plot
        Y = batch_layer_computation(X, W, b)
        Y_test = batch_layer_computation(X_test, W, b)
        Y_valid = batch_layer_computation(X_valid, W, b)
        data = calculate_data(Y, T, Y_test, T_test, Y_valid, T_valid)
        # gradient descent procedure
        summary = np.hstack((summary, data[:,None]))
        p = mu*p + alpha*CE_dWeight(X, Y, T)
        W = W - p
        b = b - alpha*CE_dBias(Y, T)
        # print some information
        if count % 5 == 0:
            print "Iter: " , count
            print "Weight: " , W
            print "Cost: " , part3_cross_entropy(Y, T)
            print "Accuracy on train set: ", calculate_accuracy(Y, T)
            print "Accuracy on valid set: ", calculate_accuracy(Y_valid, T_valid)
        count += 1
    Y = batch_layer_computation(X, W, b)
    print "Final Perfromance is: " + str(calculate_accuracy(Y_test, T_test))
    if mu == 0:
        np.save('tmp/part4_summary_data'+str(alpha), summary)
    else:
        np.save('tmp/part4_summary_data_momentum'+str(alpha), summary)
    return W, b, summary


def part6_dfW(X, Y, T, coord1, coord2):
    dW = CE_dWeight(X, Y, T)
    dw1 = dW[coord1[0]][coord1[1]]
    dw2 = dW[coord2[0]][coord2[1]]
    dw1w2 = np.zeros(dW.shape)
    dw1w2[coord1[0]][coord1[1]] = dw1
    dw1w2[coord2[0]][coord2[1]] = dw2
    return dw1w2

def get_w1_w2(W, coord1, coord2):
    return W[coord1[0]][coord1[1]], W[coord2[0]][coord2[1]]



def gradient_descent_part6(X, W, b, T, df_W, coord1, coord2, alpha=0.00001, EPS = 1e-5, max_iter = 300, mu = 0):
    """ X is the matrix of inputs, NxM, N is the number of input units, M is the 
    number of training cases. T is the matrix of results, size: LxM, L is the 
    number of output units """
    # Initialize a weight matrix
    size = [X.shape[0], T.shape[0]]
    previous_W = W - 10*EPS
    count = 0
    init_w1, init_w2 = get_w1_w2(W, coord1, coord2)
    traj = [(init_w1, init_w2)]
    p = 0
    while norm(W - previous_W) > EPS and count < max_iter:
        if count > max_iter/2:
            alpha = 0.00001
        previous_W = W
        Y = batch_layer_computation(X, W, b)
        p = mu*p + alpha*df_W(X, Y, T, coord1, coord2)
        W = W - p
        w1, w2 = get_w1_w2(W, coord1, coord2)
        traj.append((w1, w2))
        print "Iter: " , count
        print "Weight: " , W
        print "Cost: " , part3_cross_entropy(Y, T)
        if count % (max_iter/5) == 0:
            print "Accuracy on test set: ", calculate_accuracy(Y, T)
        count += 1
    return traj

def part6_func(X, W, b, T, w1, w2):
    W_size = W.shape
    W1 = np.zeros(W_size)
    W2 = np.zeros(W_size)
    W1[299][8] = 1
    W2[399][8] = 1
    return np.array([part3_cross_entropy(batch_layer_computation(X, W + w1_entry*W1 + w2_entry*W2, b), T)
        for w1_entry in w1 for w2_entry in w2]).reshape((w1.shape[0], w2.shape[0]))

def par6_get_lost(X, W, b, T, w1, w2, coord1, coord2):
    W_size = W.shape
    W1 = np.zeros(W_size)
    W2 = np.zeros(W_size)
    W1[coord1[0]][coord1[1]] = 1
    W2[coord2[0]][coord2[1]] = 1
    return part3_cross_entropy(batch_layer_computation(X, W + w1*W1 + w2*W2, b), T)

# =================================== Part 1 ===================================
if not os.path.exists('tmp') or not os.path.isdir('tmp'):
        os.mkdir('tmp')
if not os.path.exists('report') or not os.path.isdir('report'):
        os.mkdir('report')
M = loadmat("mnist_all.mat")
train_key = 'train'
for k in range(10):
    train_set = M[train_key+str(k)]
    a = np.random.randint(train_set.shape[0], size = 10)
    f, axarr = plt.subplots(5, 2)
    for i in range(5):
        for j in range(2):
            axarr[i, j].imshow(train_set[i*2+j].reshape((28,28)), cmap = cm.gray)
            axarr[i, j].axis('off')
    f.subplots_adjust(hspace=0.3)
    f.savefig('report/'+train_key+str(k)+'.png')
plt.show()


# ===================================  Part 2 ===================================
# See function layer_computation and batch_layer_computation


# ===================================  Part 3 ===================================
# See function above function

# ===================================  Part 4 ===================================
M = loadmat("mnist_all.mat")
# Initialize input X and Target matrix
X = np.empty((784, 0), dtype=np.float)
T = np.empty((10, 0), dtype=np.int)
for i in range(10):
    X_i, T_i = initialize_targets(M, 'train'+str(i), i)
    X = np.hstack((X, X_i))
    T = np.hstack((T, T_i))

for lr in [1e-4, 5e-5, 1e-5]:
    W, b, summary = gradient_descent(X, T, alpha=lr)
    train_performs = summary[0, [i*10 for i in range(300/10)]]
    test_performs = summary[1, [i*10 for i in range(300/10)]]
    valid_performs = summary[2, [i*10 for i in range(300/10)]]
    iter_arrs = [i*10 for i in range(300/10)]
    plt.plot(iter_arrs, train_performs, 'b', label='train')
    plt.plot(iter_arrs, test_performs, 'g', label='test')
    plt.plot(iter_arrs, valid_performs, 'r', label='validate')
    plt.xlabel('iteration')
    plt.ylabel('Performance')
    plt.title("Performance on lr = " + str(lr))
    plt.legend()
    plt.savefig('report/part4_perform_'+str(lr)+'.png')
    plt.show()
    train_cost = summary[3, [i*10 for i in range(300/10)]]
    test_cost = summary[4, [i*10 for i in range(300/10)]]
    valid_cost = summary[5, [i*10 for i in range(300/10)]]
    plt.plot(iter_arrs, train_cost, 'b', label='train')
    plt.plot(iter_arrs, test_cost, 'g', label='test')
    plt.plot(iter_arrs, valid_cost, 'r', label='validate')
    plt.xlabel('iteration')
    plt.ylabel('Cost')
    plt.title("Cost on lr = " + str(lr))
    plt.savefig('report/part4_cost_' + str(lr)+'.png')
    plt.legend()
    plt.show()
    for i in range(W.shape[1]):
        we = W[:, i]
        fig = figure(1)
        ax = fig.gca()    
        heatmap = ax.imshow(we.reshape((28,28)), cmap = cm.gray)
        fig.savefig('report/part4_wt_'+str(i)+'lr_'+str(lr)+'.png')
        show()
# ===================================  Part 5 ===================================
W2, b2, summary2 = gradient_descent(X, T, alpha=5e-5, mu=0.9)
iter_arrs = np.array([i for i in range(300)])
plt.plot(iter_arrs, summary[0], 'b', label='train')
plt.plot(iter_arrs, summary[1], 'g', label='test')
plt.plot(iter_arrs, summary[2], 'r', label='validate')
plt.plot(iter_arrs, summary2[0], 'c', label='train_moment')
plt.plot(iter_arrs, summary2[1], 'm', label='test_moment')
plt.plot(iter_arrs, summary2[2], 'y', label='validation_moment')
plt.xlabel('iteration')
plt.ylabel('Performance')
plt.title("Performance on lr = " + str(lr))
plt.legend()
plt.savefig('report/part5_perform_'+str(lr)+'.png')
plt.show()
plt.plot(iter_arrs, summary[3], 'b', label='train')
plt.plot(iter_arrs, summary[4], 'g', label='test')
plt.plot(iter_arrs, summary[5], 'r', label='validate')
plt.plot(iter_arrs, summary2[3], 'c', label='train_moment')
plt.plot(iter_arrs, summary2[4], 'm', label='test_moment')
plt.plot(iter_arrs, summary2[5], 'y', label='validation_moment')
plt.xlabel('iteration')
plt.ylabel('Cost')
plt.title("Cost on lr = " + str(lr))
plt.legend()
plt.savefig('report/part5_cost_' + str(lr)+'.png')
plt.show()
np.save('tmp/part5_weight_5e-5_moment', W2)
np.save('tmp/part5_bias_5e-5_moment', b2)
# ===================================  Part 6 ===================================

W = np.load('tmp/part5_weight_5e-5_moment.npy')
b = np.load('tmp/part5_bias_5e-5_moment.npy')
# W is in size NxL, we choose 9th column and 300th and 400th entry in that column
# In [14]: W[299][8]
# Out[14]: 0.21048654518881504
# In [15]: W[399][8]
# Out[15]: -0.8951020647585853
W[299][8] = 0.0
W[399][8] = 0.0
w1 = np.arange(-1., 1., 0.1)
w2 = np.arange(-1., 1., 0.1)
w1z, w2z = np.meshgrid(w1, w2)
# it takes a while to calculate 
C = np.zeros([w1.size, w2.size])
print "Calculating Contour, totally 400 iterations.............."
for i, w1_ in enumerate(w1):
    for j, w2_ in enumerate(w2):
        print "Doing {}th iteration".format(str(i*20+j))
        C[i,j] = par6_get_lost(X, W, b, T, w1_, w2_, (299, 8), (399, 8))
CS = plt.contour(w2z, w1z, C)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('w1')
plt.ylabel('w2')
plt.title("Contour for w1 as W[299][8] and w2 as W[399][8]")
plt.savefig('report/part6_parta2.png')
plt.show()

# In [13]: W[405][8]
#Out[130]: 0.557683433173424
# In [14]: W[406][8]
#Out[127]: 0.30836687545091596


W = np.load('tmp/part5_weight_5e-5_moment.npy')
b = np.load('tmp/part5_bias_5e-5_moment.npy')
W[405][8] = 1.
W[406][8] = 1.
gd_traj = gradient_descent_part6(X, W, b, T, part6_dfW, (405, 8), (406, 8),alpha=5e-5)
mo_traj = gradient_descent_part6(X, W, b, T, part6_dfW, (405, 8), (406, 8), alpha=5e-5, mu=0.9)
w1s = np.arange(-1, 1, 0.1)
w2s = np.arange(-1, 1, 0.1)
w1z, w2z = np.meshgrid(w1s, w2s)
C = np.zeros([w1s.size, w2s.size])
W[405][8] = 0
W[406][8] = 0
print "Calculating Contour, takes a while, totally 400 iters..........."
for i, w1 in enumerate(w1s):
    for j, w2 in enumerate(w2s):
        print "Doing {}th iterations".format(str(i*20+j))
        C[i,j] = par6_get_lost(X, W, b, T, w1, w2, (405, 8), (406, 8))
CS = plt.contour(w1z, w2z, C)
plt.plot([a for a, c in gd_traj], [c for a,c in gd_traj], 'yo-', label="No Momentum")
plt.plot([a for a, c in mo_traj], [c for a,c in mo_traj], 'go-', label="Momentum")
plt.xlabel('w1')
plt.ylabel('w2')
plt.clabel(CS, inline=1, fontsize=10)
plt.legend(loc='upper left')
plt.title('Contour plot')
plt.savefig('report/part6_partbc1.png')
plt.show()


#W[567][8]
#Out[129]: 0.7161147163092665
#W[406][8]
#Out[127]: 0.30836687545091596
W = np.load('tmp/part5_weight_5e-5_moment.npy')
b = np.load('tmp/part5_bias_5e-5_moment.npy')
W[326][8] = 1.
W[406][8] = 1.
gd_traj = gradient_descent_part6(X, W, b, T, part6_dfW, (326, 8), (406, 8),alpha=5e-5)
mo_traj = gradient_descent_part6(X, W, b, T, part6_dfW, (326, 8), (406, 8), alpha=5e-5, mu=0.9)
w1s = np.arange(-1, 1, 0.1)
w2s = np.arange(-1, 1, 0.1)
w1z, w2z = np.meshgrid(w1s, w2s)
C = np.zeros([w1s.size, w2s.size])
W[326][8] = 0
W[406][8] = 0
print "Calculating Contour, takes a while, totally 400 iters..........."
for i, w1 in enumerate(w1s):
    for j, w2 in enumerate(w2s):
        print "Doing {}th iterations".format(str(i*20+j))
        C[i,j] = par6_get_lost(X, W, b, T, w1, w2, (326, 8), (406, 8))
CS = plt.contour(w1z, w2z, C)
plt.plot([a for a, c in gd_traj], [c for a,c in gd_traj], 'yo-', label="No Momentum")
plt.plot([a for a, c in mo_traj], [c for a,c in mo_traj], 'go-', label="Momentum")
plt.xlabel('w1')
plt.ylabel('w2')
plt.clabel(CS, inline=1, fontsize=10)
plt.legend(loc='upper left')
plt.title('Contour plot')
plt.savefig('report/part6_partbc2.png')
plt.show()