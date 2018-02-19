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

def batch_layer_computation():
	raise NotImplementedError

def part3_cross_entropy(predicts, targets):
	"""predicts and targets are matrices of size MxL, M is the nubmer training
	cases, L is the number of output units. we should transpose the matrix 
	before we pass the argument""" 
	total = 0
	for predict, target in zip(predicts, targets):
		cost = -np.dot(target.T, np.log(predict))
		total += cost
	return total

def CE_dWeight(X, Y, T):
	"""X is the matrix of training sets with size NxM, N is the number of input
	units, M is the number of training cases, Y and T both has size LxM, L is the
	number of output layers"""
	return np.dot(X, (Y-T).T)

def approx_CE_dWeight_single(X, W, p, q, t):
	"""Approximate the derivative of single entry in Jacobian matrix"""


def approx_CE_dWeight(W):

		

# =================================== Part 1 ===================================
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