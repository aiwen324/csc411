import os
import urllib
import hashlib
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
from get_image import *
from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn



def generate_matrix(path, img_size):
    data_matrix = np.empty((0, img_size*img_size), dtype=float)
    for filename in os.listdir(path):
        im = imread(path+filename)
        im = np.true_divide(im, 255)
        im = im.flatten()
        data_matrix = np.vstack((data_matrix, im))
    return data_matrix


def generate_dataset():
    data_dict = dict()
    for dirname in os.listdir('cropped'):
        path = 'cropped/' + dirname + '/'
        if '32' in dirname:
            data_matrix = generate_matrix(path, 32)
        elif '64' in dirname:
            data_matrix = generate_matrix(path, 64)
        else:
            raise NotImplementedError
        # We split the data_matrix to 3 parts, training set, test set, validation set
        np.random.seed(0)
        matrix_idx = np.random.permutation(range(data_matrix.shape[0]))
        # print matrix_idx
        data_matrix = np.array(data_matrix[matrix_idx])
        # print data_matrix.shape
        data_size = data_matrix.shape[0]
        train_size = int((data_size-20)*0.9)
        test_set = data_matrix[:20, :]
        train_set = data_matrix[20:20+train_size, :]
        valid_set = data_matrix[20+train_size:, :]
        data_dict['test_'+dirname] = test_set
        data_dict['train_'+dirname] = train_set
        data_dict['valid_'+dirname] = valid_set
    return data_dict

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

def calculate_data(model, x_test, x_valid, test_y, valid_y, y_test_classes, y_valid_classes, loss_fn):
    y_test_pred = model(x_test)
    y_test_loss = loss_fn(y_test_pred, y_test_classes).data[0]
    y_test_pred = y_test_pred.data.numpy()
    y_test_perform = np.mean(np.argmax(y_test_pred, 1) == np.argmax(test_y, 1))
    y_valid_pred = model(x_valid)
    y_valid_loss = loss_fn(y_valid_pred, y_valid_classes).data[0]
    y_valid_pred = y_valid_pred.data.numpy()
    y_valid_perform = np.mean(np.argmax(y_valid_pred, 1) == np.argmax(valid_y, 1))
    return y_test_perform, y_valid_perform, y_test_loss, y_valid_loss

def training(lrs, batch_num, img_size, acts, data_dict, training_times=5000, hidden_units_num=512, sv_flg=1):
    train_x, train_y = get_set(data_dict, "train", img_size, acts)
    test_x, test_y = get_set(data_dict, 'test', img_size, acts)
    valid_x, valid_y = get_set(data_dict, 'valid', img_size, acts)
    # Setting up dimension
    dim_x = img_size*img_size
    dim_h = hidden_units_num
    dim_out = 6
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)

    x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
    y_test_classes = Variable(torch.from_numpy(np.argmax(test_y, 1)), requires_grad=False).type(dtype_long)

    x_valid = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
    y_valid_classes = Variable(torch.from_numpy(np.argmax(valid_y, 1)), requires_grad=False).type(dtype_long)

    mini_batch_num = batch_num
    mini_batch_size = train_x.shape[0]/mini_batch_num
    print "Number of mini-batches: ", mini_batch_num
    print "Size of mini-batches: ", mini_batch_size

    # Setting up mini-batches
    lst = []
    for i in range(mini_batch_num):
        xi = x[i*mini_batch_size:(i+1)*mini_batch_size,:]
        yi_classes = y_classes[i*mini_batch_size:(i+1)*mini_batch_size]
        yi_train = train_y[i*mini_batch_size:(i+1)*mini_batch_size]
        lst.append((xi, yi_classes, yi_train))

    torch.manual_seed(0)
    model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
    print "Resolution: {}x{}, doing gradient descent with learning rate: {} ......".format(img_size, img_size, learning_rate)
    summary = np.empty((0, 4), dtype=float)
    for t in range(training_times):
        for i in range(mini_batch_num):
            x = lst[i][0]
            y_classes = lst[i][1]
            y_pred = model(x)
            sub_train_y = lst[i][2]
            loss = loss_fn(y_pred, y_classes)
            y_pred_numpy = y_pred.data.numpy()
            y_train_perform = np.mean(np.argmax(y_pred_numpy, 1) == np.argmax(sub_train_y, 1))
            y_test_perform, y_valid_perform, y_test_loss, y_valid_loss = calculate_data(model, x_test, x_valid, test_y, valid_y,
                                                                                        y_test_classes, y_valid_classes, loss_fn)
            if t % 50 == 0:
                print "Iteration: " + str(t)
                print "loss is: " + str(loss.data[0])
                print "train_perform: ", y_train_perform
                print "valid_perform: ", y_valid_perform
            data = np.array([y_test_perform, y_valid_perform, y_test_loss, y_valid_loss])
            summary = np.vstack((summary, data))
            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to
            # make a step
    y_pred = model(x_test).data.numpy()
    print "Final performance on test set: " + str(np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1)))
    test_performs = summary[:training_times, 0]
    valid_performs = summary[:training_times, 1]
    test_loss = summary[:training_times, 2]
    valid_loss = summary[:training_times, 3]
    iter_arrs = np.arange(training_times)
    plt.plot(iter_arrs, test_performs, 'g', label='test')
    plt.plot(iter_arrs, valid_performs, 'r', label='validate')
    plt.xlabel('iteration')
    plt.ylabel('Performance')
    plt.title("Performance on lr = " + str(lrs))
    plt.legend()
    if sv_flg == 1:
        plt.savefig('report/part8'+str(img_size)+'_minibatch_performance'+str(learning_rate)+'.png')
    plt.show()
    plt.plot(iter_arrs, test_loss, 'g', label='test')
    plt.plot(iter_arrs, valid_loss, 'r', label='validate')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title("Loss on lr = " + str(lrs))
    plt.legend()
    if sv_flg == 1:
        plt.savefig('report/part8'+str(img_size)+'_minibatch_loss'+str(learning_rate)+'.png')
    plt.show()
    return model
    

# ======================= Initialization ===================
print "Please run get_image first to download image and generate info of correct images"
if not os.path.exists('tmp') or not os.path.isdir('tmp'):
    os.mkdir('tmp')
if not os.path.exists('report') or not os.path.isdir('report'):
    os.mkdir('report')
if not os.path.exists('cropped') or not os.path.isdir('cropped'):
    os.mkdir('cropped')
conver_img([32, 32])
conver_img([64, 64])

# ======================= Part 8 ===========================
print "Extracting data from cropped image..................."
data_dict = generate_dataset()
acts = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
for learning_rate in [1e-3, 1e-4, 1e-5]:
    training(learning_rate, 2, 32, acts, data_dict, training_times=4000, hidden_units_num=64)

for learning_rate in [1e-4, 1e-5]:
    training(learning_rate, 2, 64, acts, data_dict, hidden_units_num=32)

## ======================= Part 9 ===========================
model = training(1e-4, 1, 64, acts, data_dict, hidden_units_num=32, sv_flg=0)
Weights2 = model[2].weight.data.numpy()
bracco_weights = Weights2[5]
bracco_highest = np.argmax(bracco_weights)
Weights = model[0].weight.data.numpy()
bracco_highest_weight = Weights[bracco_highest]
fig = figure(1)
ax = fig.gca()    
heatmap = ax.imshow(bracco_highest_weight.reshape((64,64)), cmap = cm.coolwarm)
fig.colorbar(heatmap, shrink = 0.5, aspect=5)
fig.savefig('report/part9_64_'+acts[5]+'.png')
show()
gilpin_weights = Weights2[1]
gilpin_highest = np.argmax(gilpin_weights)
gilpin_highest_weight = Weights[gilpin_highest]
fig = figure(1)
ax = fig.gca()    
heatmap = ax.imshow(gilpin_highest_weight.reshape((64,64)), cmap = cm.coolwarm)
fig.colorbar(heatmap, shrink = 0.5, aspect=5)
fig.savefig('report/part9_64_'+acts[1]+'.png')
show()