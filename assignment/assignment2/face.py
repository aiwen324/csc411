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
# ======================= Part 8 ===========================
acts = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
# train_x, train_y = get_set(data_dict, "train", 32, acts)
# test_x, test_y = get_set(data_dict, 'test', 32, acts)
# valid_x, valid_y = get_set(data_dict, 'valid', 32, acts)
# # Setting up dimension
# dim_x = 32*32
# dim_h = 12
# dim_out = 6
# dtype_float = torch.FloatTensor
# dtype_long = torch.LongTensor

# x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
# y_classes = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)

# x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
# y_test_classes = Variable(torch.from_numpy(np.argmax(test_y, 1)), requires_grad=False).type(dtype_long)

# x_valid = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
# y_valid_classes = Variable(torch.from_numpy(np.argmax(valid_y, 1)), requires_grad=False).type(dtype_long)

# for learning_rate in [1e-2, 1e-3, 1e-4]:
#     torch.manual_seed(0)
#     model = torch.nn.Sequential(
#         torch.nn.Linear(dim_x, dim_h),
#         torch.nn.ReLU(),
#         torch.nn.Linear(dim_h, dim_out),
#     )
#     loss_fn = torch.nn.CrossEntropyLoss()
#     print "Resolution: 32x32, doing gradient descent with learning rate: {} ......".format(learning_rate)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     summary = np.empty((0, 6), dtype=float)
#     for t in range(10000):
#         y_pred = model(x)
#         loss = loss_fn(y_pred, y_classes)
#         y_pred_numpy = y_pred.data.numpy()
#         y_train_perform = np.mean(np.argmax(y_pred_numpy, 1) == np.argmax(train_y, 1))
#         y_test_perform, y_valid_perform, y_test_loss, y_valid_loss = calculate_data(model, x_test, x_valid, test_y, valid_y,
#                                                                                     y_test_classes, y_valid_classes)
#         data = np.array([y_train_perform, y_test_perform, y_valid_perform, loss.data[0], y_test_loss, y_valid_loss])
#         summary = np.vstack((summary, data))
#         model.zero_grad()  # Zero out the previous gradient computation
#         loss.backward()    # Compute the gradient
#         optimizer.step()   # Use the gradient information to
#         # make a step
#     y_pred = model(x_test).data.numpy()
#     print np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
#     np.save('tmp/part8_summary'+str(learning_rate), summary)
#     train_performs = summary[:5000, 0]
#     test_performs = summary[:5000, 1]
#     valid_performs = summary[:5000, 2]
#     train_loss = summary[:5000, 3]
#     test_loss = summary[:5000, 4]
#     valid_loss = summary[:5000, 5]
#     iter_arrs = np.arange(5000)
#     plt.plot(iter_arrs, train_performs, 'b', label='train')
#     plt.plot(iter_arrs, test_performs, 'g', label='test')
#     plt.plot(iter_arrs, valid_performs, 'r', label='validate')
#     plt.xlabel('iteration')
#     plt.ylabel('Performance')
#     plt.legend()
#     plt.savefig('report/part8_curves_performance'+str(learning_rate)+'.png')
#     plt.show()
#     plt.plot(iter_arrs, train_loss, 'b', label='train')
#     plt.plot(iter_arrs, test_loss, 'g', label='test')
#     plt.plot(iter_arrs, valid_loss, 'r', label='validate')
#     plt.xlabel('iteration')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.savefig('report/part8_curves_loss'+str(learning_rate)+'.png')
#     plt.show()

# train_x, train_y = get_set(data_dict, "train", 64, acts)
# test_x, test_y = get_set(data_dict, 'test', 64, acts)
# valid_x, valid_y = get_set(data_dict, 'valid', 64, acts)
# # Setting up dimension
# dim_x = 64*64
# dim_h = 12
# dim_out = 6
# dtype_float = torch.FloatTensor
# dtype_long = torch.LongTensor

# x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
# y_classes = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)

# x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
# y_test_classes = Variable(torch.from_numpy(np.argmax(test_y, 1)), requires_grad=False).type(dtype_long)

# x_valid = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
# y_valid_classes = Variable(torch.from_numpy(np.argmax(valid_y, 1)), requires_grad=False).type(dtype_long)

# for learning_rate in [1e-4, 1e-5, 1e-6]:
#     torch.manual_seed(0)
#     model = torch.nn.Sequential(
#         torch.nn.Linear(dim_x, dim_h),
#         torch.nn.ReLU(),
#         torch.nn.Linear(dim_h, dim_out),
#     )
#     loss_fn = torch.nn.CrossEntropyLoss()
#     print "Resolution: 64x64, doing gradient descent with learning rate: {} ......".format(learning_rate)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     summary = np.empty((0, 6), dtype=float)
#     for t in range(10000):
#         y_pred = model(x)
#         loss = loss_fn(y_pred, y_classes)
#         y_pred_numpy = y_pred.data.numpy()
#         y_train_perform = np.mean(np.argmax(y_pred_numpy, 1) == np.argmax(train_y, 1))
#         y_test_perform, y_valid_perform, y_test_loss, y_valid_loss = calculate_data(model, x_test, x_valid, test_y, valid_y,
#                                                                                     y_test_classes, y_valid_classes)
#         data = np.array([y_train_perform, y_test_perform, y_valid_perform, loss.data[0], y_test_loss, y_valid_loss])
#         summary = np.vstack((summary, data))
#         model.zero_grad()  # Zero out the previous gradient computation
#         loss.backward()    # Compute the gradient
#         optimizer.step()   # Use the gradient information to
#         # make a step
#     y_pred = model(x_test).data.numpy()
#     print np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
#     np.save('tmp/part8_summary'+str(learning_rate), summary)
#     train_performs = summary[:10000, 0]
#     test_performs = summary[:10000, 1]
#     valid_performs = summary[:10000, 2]
#     train_loss = summary[:10000, 3]
#     test_loss = summary[:10000, 4]
#     valid_loss = summary[:10000, 5]
#     iter_arrs = np.arange(10000)
#     plt.plot(iter_arrs, train_performs, 'b', label='train')
#     plt.plot(iter_arrs, test_performs, 'g', label='test')
#     plt.plot(iter_arrs, valid_performs, 'r', label='validate')
#     plt.xlabel('iteration')
#     plt.ylabel('Performance')
#     plt.legend()
#     plt.savefig('report/part8_64curves_performance'+str(learning_rate)+'.png')
#     plt.show()
#     plt.plot(iter_arrs, train_loss, 'b', label='train')
#     plt.plot(iter_arrs, test_loss, 'g', label='test')
#     plt.plot(iter_arrs, valid_loss, 'r', label='validate')
#     plt.xlabel('iteration')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.savefig('report/part8_64curves_loss'+str(learning_rate)+'.png')
#     plt.show()


# ======================= Part 9 ===========================
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

learning_rate = 0.0001
torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)
loss_fn = torch.nn.CrossEntropyLoss()
print "Resolution: 32x32, doing gradient descent with learning rate: {} ......".format(learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    
    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to
    # make a step
y_pred = model(x_test).data.numpy()
print np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
Weights = model[0].weight.data.numpy()
for Weight in Weights:
    fig = figure(1)
    ax = fig.gca()    
    heatmap = ax.imshow(Weight.reshape((32,32)), cmap = cm.coolwarm)
    fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    show()

# ======================= Part 10 ===========================
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256)
            nn.ReLU()
            nn.Linear(256, 6)
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

