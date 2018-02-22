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

male_actors = ['baldwin', 'hader', 'radcliffe', 'butler', 'vartan', 'carell']
actresses = ['bracco', 'chenoweth', 'drescher', 'ferrera', 'gilpin', 'harmon']
male_actors_fullname = ['Alec Baldwin', 'Bill Hader', 'Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Steve Carell']
actresses_fullname = ['Lorraine Bracco', 'Kristin Chenoweth', 'Fran Drescher', 'America Ferrera', 'Peri Gilpin', 'Angie Harmon']
test_actors = ['Alec Baldwin']


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray / 255.


def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    	f.close()
    return sha256.hexdigest()


# testfile = urllib.URLopener()
# os.mkdir('uncropped')
# for a in actresses_fullname:
#     # Parsing Lorraine Bracco
#     name = a.split()[1].lower()
#     if not os.path.exists("uncropped/"+name) or not os.path.isdir("uncropped/"+name):
#         os.mkdir("uncropped/"+name)
#     fname = str(name) + ".txt"
#     f = open(fname, "w")
#     i = 0
#     for line in open("facescrub_actresses.txt"):
#         if a in line:
#             filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
#             #A version without timeout (uncomment in case you need to 
#             #unsupress exceptions, which timeout() does)
#             #testfile.retrieve(line.split()[4], "uncropped/"+filename)
#             #timeout is used to stop downloading images which take too long to download
#             filepath = "uncropped/"+name+"/"+filename
#             timeout(testfile.retrieve, (line.split()[4], filepath), {}, 30)
#             if not os.path.isfile("uncropped/"+name+"/"+filename):
#                 continue
#             hashword = sha256_checksum(filepath)
#             if hashword != line.split()[6]:
#             	os.remove(filepath)
#             	print "Detecting unmatched HASH!!!!!!"
#             	continue
#             print filename
#             line_to_wrt = line.split()[1] + " " + filename + "\t" + line.split()[4] + "\t" + line.split()[5] + '\n'
#             f.write(line_to_wrt)
#             i += 1
#     f.close()

# for a in male_actors_fullname:
#     name = a.split()[1].lower()
#     if not os.path.exists("uncropped/"+name) or not os.path.isdir("uncropped/"+name):
#         os.mkdir("uncropped/"+name)
#     fname = str(name) + ".txt"
#     f = open(fname, "w")
#     i = 0
#     for line in open("facescrub_actors.txt"):
#         if a in line:
#             filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
#             #A version without timeout (uncomment in case you need to 
#             #unsupress exceptions, which timeout() does)
#             #testfile.retrieve(line.split()[4], "uncropped/"+filename)
#             #timeout is used to stop downloading images which take too long to download
#             filepath = "uncropped/"+name+"/"+filename
#             timeout(testfile.retrieve, (line.split()[4], filepath), {}, 30)
#             if not os.path.isfile("uncropped/"+name+"/"+filename):
#                 continue
#             print filename
#             hashword = sha256_checksum(filepath)
#             if hashword != line.split()[6]:
#             	os.remove(filepath)
#             	print "Detecting unmatched HASH!!!!!!"
#             	continue
#             line_to_wrt = line.split()[1] + " " + filename + "\t" + line.split()[4] + "\t" + line.split()[5] + '\n'
#             f.write(line_to_wrt)
#             i += 1
#     f.close()

def conver_img(img_size):
    for actor in os.listdir("uncropped"):
        act_bd_box_file = open(actor + ".txt")
        img_path = 'cropped/' + actor + str(img_size[0]) 
        if not os.path.exists(img_path) or not os.path.isdir(img_path):
            os.mkdir(img_path)
        for filename in os.listdir("uncropped/" + actor):
            print ("dealing with " + filename)
            try:
                im = imread("uncropped/" + actor + "/" + filename)
                # move file pointer to 0
                act_bd_box_file.seek(0)
                for line in act_bd_box_file:
                    if filename in line:
                        print ("find match for " + filename)
                        bd_box = line.split()[3].split(',')
                        bd_box = [int(a) for a in bd_box]
                        cropped_im = im[bd_box[1]:bd_box[3], bd_box[0]:bd_box[2]]
                        # Check if we the image is 2D or 3D
                        if len(cropped_im.shape) != 2:
                            cropped_im = rgb2gray(cropped_im)
                        # Resize the image
                        cropped_im = imresize(cropped_im, img_size)
                        imsave(img_path + "/" + filename, cropped_im)
                        break
            # If detect some image cannot be opened in the system, report error
            except IOError:
                f = open(actor + "_err.txt", "a")
                f.write(filename + '\n')
                f.close()
                
                
def conver_img2(img_size):
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
                
def generate_matrix2(path):
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

def generate_dataset2():
    data_dict = dict()
    for dirname in os.listdir('cropped2'):
        path = 'cropped2/'+ dirname + '/'
        data_matrix = generate_matrix2(path)
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