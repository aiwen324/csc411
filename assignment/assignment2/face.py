import os
import urllib
import hashlib
from rgb2gray import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from scipy.ndimage import filters
from shutil import copy2
import pickle
from get_image import *
from torch.autograd import Variable
import torch


def get_set(M, set_type, img_size, acts):
    batch_xs = np.empty((0, img_size*img_size))
    batch_y_s = np.empty((0, len(acts)))
    
    train_k =  [set_type+"_"+act+str(img_size) for act in acts]
    for k in range(len(acts)):
        batch_xs = np.vstack((batch_xs, M[train_k[k]]))
        one_hot = np.zeros(len(acts))
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (M[train_k[k]].shape[0], 1))))
    return batch_xs, batch_y_s

def calculate_performance(Y, T):
    """ Calculate the performance , Y and T a matrices in size Nx6"""
    return np.mean(np.argmax(Y, axis=1) == np.argmax(T, axis=1))


# ======================= Initialization ===================
#if not os.path.exists('tmp'):
#    os.mkdir('tmp')
#os.mkdir('cropped')
#conver_img([32, 32])
#conver_img([64, 64])
#data_dict = generate_dataset()
#pickle_out = open('tmp/part8_dataset.pickle', "w")
#cPickle.dump(data_dict, pickle_out)
#pickle_out.close()
# ======================= Part 8 ===========================
acts = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
data_dict = cPickle.load(open('tmp/part8_dataset.pickle'))
train_x, train_y = get_set(data_dict, "train", 32, acts)
test_x, test_y = get_set(data_dict, 'test', 32, acts)
valid_x, valid_batch_y = get_set(data_dict, 'valid', 32, acts)
dim_x = 32*32
dim_h = 12
dim_out = 6
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)
x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_long)
x_train = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_long)
torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
summary = np.empty((6, 0), dtype=float)
for t in range(10000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    
    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step