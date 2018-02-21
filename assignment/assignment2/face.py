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
import pickle
import cPickle
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

def calculate_data(model, x_test, x_valid, test_y, valid_y, y_test_classes, y_valid_classes):
    y_test_pred = model(x_test)
    y_test_loss = loss_fn(y_test_pred, y_test_classes).data[0]
    y_test_pred = y_test_pred.data.numpy()
    y_test_perform = np.mean(np.argmax(y_test_pred, 1) == np.argmax(test_y, 1))
    y_valid_pred = model(x_valid)
    y_valid_loss = loss_fn(y_valid_pred, y_valid_classes).data[0]
    y_valid_pred = y_valid_pred.data.numpy()
    y_valid_perform = np.mean(np.argmax(y_valid_pred, 1) == np.argmax(valid_y, 1))
    return y_test_perform, y_valid_perform, y_test_loss, y_valid_loss



# ======================= Initialization ===================
#if not os.path.exists('tmp'):
#    os.mkdir('tmp')
#os.mkdir('cropped')
# conver_img([32, 32])
# conver_img([64, 64])
data_dict = generate_dataset()
pickle_out = open('tmp/part8_dataset.pickle', "w")
cPickle.dump(data_dict, pickle_out)
pickle_out.close()
# ======================= Part 8 ===========================
acts = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
data_dict = cPickle.load(open('tmp/part8_dataset.pickle'))
train_x, train_y = get_set(data_dict, "train", 32, acts)
test_x, test_y = get_set(data_dict, 'test', 32, acts)
valid_x, valid_y = get_set(data_dict, 'valid', 32, acts)
# Setting up dimension
dim_x = 32*32
dim_h = 12
dim_out = 6
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)

x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
y_test_classes = Variable(torch.from_numpy(np.argmax(test_y, 1)), requires_grad=False).type(dtype_long)

x_valid = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
y_valid_classes = Variable(torch.from_numpy(np.argmax(valid_y, 1)), requires_grad=False).type(dtype_long)

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
    torch.nn.Softmax()
)
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
summary = np.empty((0, 6), dtype=float)
for t in range(10000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    y_pred_numpy = y_pred.data.numpy()
    y_train_perform = np.mean(np.argmax(y_pred_numpy, 1) == np.argmax(train_y, 1))
    y_test_perform, y_valid_perform, y_test_loss, y_valid_loss = calculate_data(model, x_test, x_valid, test_y, valid_y,
                                                                                y_test_classes, y_valid_classes)
    data = np.array([y_train_perform, y_test_perform, y_valid_perform, loss.data[0], y_test_loss, y_valid_loss])
    summary = np.vstack((summary, data))
    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step
x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
y_pred = model(x).data.numpy()
np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
np.save('tmp/part8_summary'+str(learning_rate), summary)
train_performs = summary[:, 0]
test_performs = summary[:, 1]
valid_performs = summary[:, 2]
train_loss = summary[:, 3]
test_loss = summary[:, 4]
valid_loss = summary[:, 5]
iter_arrs = np.arange(summary.shape[0])
plt.plot(iter_arrs, train_performs, 'b', label='train')
plt.plot(iter_arrs, test_performs, 'g', label='test')
plt.plot(iter_arrs, valid_performs, 'r', label='validate')
plt.xlabel('iteration')
plt.ylabel('Performance')
plt.legend()
plt.savefig('report/part8_curves'+str(learning_rate)+'.png')
plt.show()


