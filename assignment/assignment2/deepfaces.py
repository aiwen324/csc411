import os
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

def conver_img(img_size):
    if not os.path.exists('cropped2') or not os.path.isdir('cropped2'):
        os.mkdir('cropped2')
    for actor in os.listdir("uncropped"):
        act_bd_box_file = open(actor + ".txt")
        img_path = 'cropped2/' + actor + str(img_size[0]) 
        if not os.path.exists(img_path) or not os.path.isdir(img_path):
            os.mkdir(img_path)
        for filename in os.listdir("uncropped/" + actor):
            try:
                im = imread("uncropped/" + actor + "/" + filename)
                # move file pointer to 0
                act_bd_box_file.seek(0)
                for line in act_bd_box_file:
                    if filename in line:
                        bd_box = line.split()[3].split(',')
                        bd_box = [int(a) for a in bd_box]
                        cropped_im = im[bd_box[1]:bd_box[3], bd_box[0]:bd_box[2]]
                        # Check if we the image is 2D or 3D
                        if len(cropped_im.shape) == 2:
                            print "Find {} is not a 3D array".format(filename)
                            continue
                        # Resize the image, here should be (227, 227, 3)
                        cropped_im = imresize(cropped_im, img_size)
                        imsave(img_path + "/" + filename, cropped_im)
                        break
            # If detect some image cannot be opened in the system, report error
            except IOError:
                f = open(actor + "_err.txt", "a")
                f.write(filename + '\n')
                f.close()

def generate_matrix(path):
    data_tensor = []
    for filename in os.listdir(path):
        try:
            im = imread(path+filename)[:, :, :3]
        except IOError as inst:
            print inst.args
            os.remove(path+filename)
            continue
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32)
        data_tensor.append(im)
    data_matrix = np.stack(data_tensor, axis=0)
#    print "Get data as shape: ", data_matrix.shape
    return data_matrix

def generate_dataset():
    data_dict = dict()
    for dirname in os.listdir('cropped2'):
        path = 'cropped2/'+ dirname + '/'
        data_matrix = generate_matrix(path)
        np.random.seed(0)
        matrix_idx = np.random.permutation(range(data_matrix.shape[0]))
        data_matrix = np.array(data_matrix[matrix_idx])
        data_size = data_matrix.shape[0]
        train_size = int((data_size-20)*0.9)
        test_set = data_matrix[:20, :, :, :]
        train_set = data_matrix[20:20+train_size, :, :, :]
        valid_set = data_matrix[20+train_size:, :, :, :]
        data_dict['test_'+dirname] = test_set
        data_dict['train_'+dirname] = train_set
        data_dict['valid_'+dirname] = valid_set
    return data_dict

def get_set(M, set_type, img_size, acts):
    batch_xs = np.empty((0, img_size[0], img_size[1], img_size[2]))
    batch_y_s = np.empty((0, len(acts)))
    
    train_k = [set_type+"_"+act+str(img_size[1]) for act in acts]
    for k in range(len(acts)):
        batch_xs = np.vstack((batch_xs, M[train_k[k]]))
        one_hot = np.zeros(len(acts))
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (M[train_k[k]].shape[0], 1))))
    return batch_xs, batch_y_s

def extract_data(x, pre_load_model):
    x = Variable(torch.from_numpy(x), requires_grad=False).type(dtype_float)
    x = pre_load_model.features(x)
    x = x.view(x.size(0), 256*6*6)
    x = x.data.numpy()
    x = Variable(torch.from_numpy(x), requires_grad=False).type(dtype_float)
    return x

# ======================= Part 10 ===========================
print "Converting image.............................."
#conver_img((227, 227, 3))
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

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
        
        self.load_weights()

# Generate data from cropped image
print "Getting data from cropped image............."
data_dict2 = generate_dataset()

torch.manual_seed(0)

pre_load_model = MyAlexNet()
pre_load_model.eval()

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
# train_size is 408 here
# Get matrix from dictionary data_dict2 with specified acts
acts = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
train_x_whole, train_y_whole = get_set(data_dict2, 'train', (3, 227, 227), acts)
test_x, test_y = get_set(data_dict2, 'test', (3,227,227), acts)
valid_x, valid_y = get_set(data_dict2, 'valid', (3,227,227), acts)
# Define the model we need to train
model_to_train = nn.Sequential(
   nn.Linear(256*6*6, 12),
   nn.ReLU(),
   nn.Linear(12, 6),
   )
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_to_train.parameters(), lr=1e-5)
# optimizer = torch.optim.SGD(model_to_train.parameters(), lr = 1e-5, momentum=0.9)


print "Extraacting data from features........"
x = extract_data(train_x_whole, pre_load_model)
y_classes = Variable(torch.from_numpy(np.argmax(train_y_whole, 1)), requires_grad=False).type(dtype_long)
x_valid = extract_data(valid_x, pre_load_model)
for t in range(10000):
    y_pred = model_to_train(x)
    loss = loss_fn(y_pred, y_classes)
    
    if t % 20 == 0:
        y_valid_pred = model_to_train(x_valid).data.numpy()
        print "Iteration " + str(t) + " loss is: " + str(loss.data[0])
        print "Performacne on valid set: " + str(np.mean(np.argmax(y_valid_pred, 1) == np.argmax(valid_y, 1)))
    model_to_train.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()


print "Calculating performacne.................."
x_test = extract_data(test_x, pre_load_model)
y_pred = model_to_train(x_test).data.numpy()
print "Performacne on test set is: " + str(np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1)))